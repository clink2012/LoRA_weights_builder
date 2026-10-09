import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildStartingProposal } from './buildStartingProposal';
import { root, source, job, proposal, digest } from './startingProposalFixture';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
describe('Main preparation orchestration', () => {
  let reused, starting, final;
  beforeEach(() => {
    reused = false; starting = 'complete'; final = proposal();
    vi.stubGlobal('fetch', vi.fn(async (url) => {
      if (url.endsWith('/defaults')) return response(root(url.split('/').at(-2)));
      if (url.endsWith('/prepare-blocks')) return response(source());
      if (url.endsWith('/resolve')) return response(reused ? { status: 'reused', job: job() } : { status: 'not_found', job: null });
      if (url.endsWith('/analysis-jobs')) return response(job(starting));
      if (url.endsWith('/cancel')) return response(job('cancelled'));
      if (url.endsWith('/experiments/prepare')) return response(final);
      return response(job());
    }));
  });
  afterEach(() => vi.unstubAllGlobals());
  const build = (options = {}) => buildStartingProposal('/api', ['A', 'B'], { signal: new AbortController().signal, onProgress: vi.fn(), ...options });
  it('captures Defaults, measures once, and returns a matching server-owned proposal', async () => {
    const result = await build();
    expect(result.node_payloads[0].loader_export.slot_values[1]).toBe(.9);
    expect(fetch.mock.calls.filter(([url]) => url.endsWith('/analysis-jobs'))).toHaveLength(1);
    const body = JSON.parse(fetch.mock.calls.find(([url]) => url.endsWith('/experiments/prepare'))[1].body);
    expect(body).toEqual({ job_id: 'measured', priorities: {}, expected_preparation_digest: digest, force_recompute: false });
    expect(fetch.mock.calls.some(([url]) => url.includes('composition-preferences'))).toBe(false);
  });
  it('reuses a matching measurement and can explicitly recompute the baseline', async () => {
    reused = true;
    await build({ force: true });
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/analysis-jobs'))).toBe(false);
    expect(JSON.parse(fetch.mock.calls.at(-1)[1].body).force_recompute).toBe(true);
  });
  it('fails without a neutral fallback when analysis is unavailable', async () => {
    fetch.mockImplementation(async (url) => url.endsWith('/defaults') ? response(root(url.split('/').at(-2))) : url.endsWith('/prepare-blocks') ? response(source()) : response({ detail: { reason: 'CPU runtime unavailable' } }, 503));
    await expect(build()).rejects.toThrow('CPU runtime unavailable');
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/experiments/prepare'))).toBe(false);
  });
  it('rejects mismatched proposed values rather than showing a graph different from Copy', async () => {
    final.node_payloads[1].loader_export.architecture_slot_values = Array(58).fill(1);
    await expect(build()).rejects.toThrow('proposed graph and loader values');
  });
  it('cancels its obsolete newly-created worker even if selection changes during the start request', async () => {
    const controller = new AbortController();
    const original = fetch.getMockImplementation();
    fetch.mockImplementation(async (url, init) => {
      if (url.endsWith('/analysis-jobs')) { controller.abort(); return response(job('running')); }
      return original(url, init);
    });
    await expect(build({ signal: controller.signal })).rejects.toThrow('Preparation cancelled');
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/measured/cancel'))).toBe(true);
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/experiments/prepare'))).toBe(false);
  });
  it('polls the worker to completion before requesting export', async () => {
    starting = 'running';
    const result = await build();
    expect(result.measurement_job.status).toBe('complete');
    expect(fetch.mock.calls.some(([url]) => url === '/api/analysis-jobs/measured')).toBe(true);
  });
});
