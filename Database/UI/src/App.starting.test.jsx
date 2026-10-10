import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { root, node, job, proposal, source, target, digest } from './studio/startingProposalFixture';

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
describe('Build a starting proposal and deliberately load recipes', { timeout: 15000 }, () => {
  let profiles, recipes, failMeasurement;
  beforeEach(() => {
    profiles = ['A', 'B'].map((id) => root(id)); recipes = []; failMeasurement = false;
    const stored = new Map(); vi.stubGlobal('localStorage', { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) });
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } });
    vi.stubGlobal('fetch', vi.fn(async (input, init) => {
      const path = new URL(String(input), 'http://localhost').pathname;
      const body = init?.body ? JSON.parse(init.body) : null;
      if (path.endsWith('/model-families')) return response({ families: [{ code: 'FLX', display_name: 'FLUX.1', support_level: 'experimental' }] });
      if (path.endsWith('/catalogue') || path.endsWith('/catalogue/compatible') || path.endsWith('/lora/search')) return response({ total: 2, results: ['A', 'B'].map((stable_id) => ({ stable_id, filename: `${stable_id}.safetensors`, base_model_code: 'FLX', role: root(stable_id).settings.role, compatibility: { status: 'eligible' } })) });
      if (path.endsWith('/library-scan')) return response({ status: 'idle' });
      if (path.endsWith('/defaults')) return response(root(path.split('/').at(-2)));
      if (path.includes('/profile-versions/') && path.endsWith('/revisions')) {
        const id = path.split('/').at(-2), parent = root(id);
        const saved = { ...parent, ...body, stable_id: id, kind: 'personal', sequence: profiles.length + 1, version_id: `personal-${id}` };
        profiles.push(saved); return response(saved);
      }
      if (path.endsWith('/selection')) return response(profiles.find((p) => p.version_id === body?.version_id) || root(path.split('/').at(-2)));
      if (path.includes('/profile-versions/') && path.includes('/versions/')) return response(profiles.find((p) => p.version_id === path.split('/').at(-1)));
      if (path.includes('/profile-versions/')) return response({ versions: profiles.filter((p) => p.stable_id === path.split('/').at(-1)) });
      if (path.endsWith('/prepare-blocks')) return response({ ...source(body.stable_ids), node_payloads: body.stable_ids.map((id) => node(profiles.find((p) => p.version_id === body.profile_version_ids?.[id]) || root(id))) });
      if (path.endsWith('/analysis-jobs/resolve')) return response({ status: 'not_found', job: null });
      if (path.endsWith('/analysis-jobs')) return failMeasurement ? response({ detail: { reason: 'CPU analysis unavailable' } }, 503) : response(job());
      if (path.endsWith('/experiments/prepare')) return response(proposal());
      if (path.endsWith('/experiments/save')) {
        const data = proposal();
        const savedProfiles = data.source_profiles.map((p, i) => ({ ...p, values: data.starting_proposal.policy_preview.entries[i].values, kind: 'personal', name: body.name, version_id: `personal-${p.stable_id}` }));
        profiles.push(...savedProfiles);
        const recipe = { version_id: 'saved-start', name: body.name, target_contract_id: target, entries: savedProfiles.map((p) => ({ stable_id: p.stable_id, profile_version_id: p.version_id })) };
        recipes.push(recipe); return response({ status: 'saved', composition: recipe });
      }
      if (path.endsWith('/composition-versions')) return response({ versions: recipes });
      if (path.includes('/composition-versions/')) return response(recipes.find((r) => r.version_id === path.split('/').at(-1)));
      return response({}, 404);
    }));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
  async function choose(id) {
    const button = await screen.findByRole('button', { name: new RegExp(`${id}.*FLX.*${id}`) });
    await waitFor(() => expect(button.disabled).toBe(false)); fireEvent.click(button);
  }
  async function build() {
    fireEvent.click(screen.getByRole('button', { name: 'Prepare block values' }));
    await screen.findByRole('region', { name: 'Full loader vectors' });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(false));
  }
  async function start() { render(<App />); await choose('A'); await choose('B'); await build(); }

  it('retains every crossed bar from a fast stroke and the peer proposal when saved', async () => {
    await start();
    const bars = [...document.querySelectorAll('.studio-measured-bars button')];
    bars.forEach((bar, i) => { bar.getBoundingClientRect = () => ({ left: i * 50, right: i * 50 + 48, bottom: 200, height: 200 }); });
    fireEvent.pointerDown(bars[1], { button: 0, buttons: 1, pointerId: 7, clientX: 74, clientY: 100 });
    fireEvent.pointerMove(bars[1], { buttons: 1, pointerId: 7, clientX: 224, clientY: 100 });
    fireEvent.pointerUp(bars[1], { pointerId: 7 });
    const expected = Number(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value);
    for (let i = 0; i < 4; i++) expect(Number(screen.getByRole('textbox', { name: `Multiplier for DOUBLE ${i}` }).value)).toBeCloseTo(expected);
    fireEvent.change(screen.getByLabelText('Revision name'), { target: { value: 'Drawn character' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save new revision' }));
    await waitFor(() => expect(profiles.some((p) => p.name === 'Drawn character')).toBe(true));
    expect(profiles.find((p) => p.name === 'Drawn character').values.slice(1, 5)).toEqual([expected, expected, expected, expected]);
    const peer = within(screen.getByRole('region', { name: 'Selected stack' })).getByRole('button', { name: /Loader 2.*clothing.*B/ });
    fireEvent.click(peer);
    expect(Number(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value)).toBe(.65);
  });

  it('one preparation produces editable role values, original lines and whole-vector Copy', async () => {
    await start();
    expect(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value).toBe('0.9');
    const vectors = screen.getByRole('region', { name: 'Full loader vectors' });
    expect(within(vectors).getByRole('textbox', { name: 'Full block values for B' }).value.split(',')[1]).toBe('0.65');
    expect(document.querySelector('.studio-original-line')).toBeTruthy();
    fireEvent.click(within(vectors).getAllByRole('button', { name: 'Copy full vector' })[0]);
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith(proposal().node_payloads[0].loader_export.numeric_csv));
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/composition-preferences/resolve'))).toBe(false);
    expect(profiles).toHaveLength(2); expect(recipes).toHaveLength(0);
  });
  it('saves the whole starting proposal only on request and revalidates loaded values without applying role cuts twice', async () => {
    await start(); fireEvent.click(screen.getByText('Saved compositions'));
    fireEvent.change(screen.getByLabelText('Recipe name'), { target: { value: 'My starting pair' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save recipe version' }));
    await waitFor(() => expect(recipes).toHaveLength(1));
    expect(screen.queryByRole('region', { name: 'Full loader vectors' })).toBeNull();
    await build();
    expect(screen.getByRole('textbox', { name: 'Full block values for A' }).value.split(',')[1]).toBe('0.9');
    expect(screen.getByRole('textbox', { name: 'Full block values for B' }).value.split(',')[1]).toBe('0.65');
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/experiments/prepare'))).toHaveLength(1);
    expect(profiles[0].values[1]).toBe(1);
    fireEvent.click(screen.getByRole('button', { name: 'Clear stack' }));
    await choose('A'); await choose('B');
    expect(screen.queryByText('My starting pair')).toBeNull();
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/composition-preferences/resolve'))).toBe(false);
  });
  it('editing preserves every proposed vector until each personal draft is saved or discarded', async () => {
    await start();
    fireEvent.change(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }), { target: { value: '.8' } });
    expect(screen.queryByRole('region', { name: 'Full loader vectors' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('Revision name'), { target: { value: 'Adjusted character' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save new revision' }));
    await waitFor(() => expect(profiles.some((p) => p.version_id === 'personal-A')).toBe(true));
    expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(true);
    fireEvent.click(within(screen.getByRole('region', { name: 'Selected stack' })).getByRole('button', { name: /Loader 2.*clothing.*B/ }));
    expect(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value).toBe('0.65');
    fireEvent.change(screen.getByLabelText('Revision name'), { target: { value: 'Starting clothing' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save new revision' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(false));
    await build();
    expect(screen.getByRole('textbox', { name: 'Full block values for A' }).value.split(',')[1]).toBe('0.8');
    expect(screen.getByRole('textbox', { name: 'Full block values for A' }).value.split(',')[2]).toBe('0.9');
    expect(screen.getByRole('textbox', { name: 'Full block values for B' }).value.split(',')[1]).toBe('0.65');
  });
  it('reports unavailable analysis without showing all-1 placeholders as a proposal', async () => {
    failMeasurement = true; render(<App />); await choose('A'); await choose('B');
    fireEvent.click(screen.getByRole('button', { name: 'Prepare block values' }));
    await screen.findByRole('alert');
    expect(screen.getByRole('alert').textContent).toContain('CPU analysis unavailable');
    expect(screen.queryByRole('region', { name: 'Full loader vectors' })).toBeNull();
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/experiments/prepare'))).toBe(false);
    const request = fetch.mock.calls.find(([url]) => String(url).endsWith('/analysis-jobs'));
    expect(JSON.parse(request[1].body).expected_preparation_digest).toBe(digest);
  });
  it('loading a history version preserves the other unsaved proposal until explicitly saved', async () => {
    const earlier = { ...root('A'), kind: 'personal', name: 'Earlier character', sequence: 2, version_id: 'earlier-A', values: root('A').values.map((value, index) => index ? .7 : value) };
    profiles.push(earlier);
    await start();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh variants & history' }));
    const history = await screen.findByRole('button', { name: /Earlier character.*Version 2/ });
    await waitFor(() => expect(history.disabled).toBe(false)); fireEvent.click(history);
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value).toBe('0.7'));
    expect(screen.queryByRole('region', { name: 'Full loader vectors' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(true);
    fireEvent.click(within(screen.getByRole('region', { name: 'Selected stack' })).getByRole('button', { name: /Loader 2.*clothing.*B/ }));
    expect(screen.getByRole('textbox', { name: 'Multiplier for DOUBLE 0' }).value).toBe('0.65');
    fireEvent.change(screen.getByLabelText('Revision name'), { target: { value: 'Preserved clothing' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save new revision' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Prepare block values' }).disabled).toBe(false));
    await build();
    expect(screen.getByRole('textbox', { name: 'Full block values for A' }).value.split(',')[1]).toBe('0.7');
    expect(screen.getByRole('textbox', { name: 'Full block values for B' }).value.split(',')[1]).toBe('0.65');
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/experiments/prepare'))).toHaveLength(1);
  });
  it('reuses the proposal save key after an uncertain response and keeps values available', async () => {
    await start(); fireEvent.click(screen.getByText('Saved compositions'));
    fireEvent.change(screen.getByLabelText('Recipe name'), { target: { value: 'Retry pair' } });
    const originalFetch = fetch.getMockImplementation();
    let interrupt = true;
    fetch.mockImplementation(async (input, init) => {
      if (String(input).endsWith('/experiments/save') && interrupt) { interrupt = false; throw new Error('Connection interrupted'); }
      return originalFetch(input, init);
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save recipe version' }));
    await screen.findByText(/The save outcome could not be confirmed/);
    expect(screen.getByRole('textbox', { name: 'Full block values for B' }).value.split(',')[1]).toBe('0.65');
    fireEvent.click(screen.getByRole('button', { name: 'Save recipe version' }));
    await waitFor(() => expect(recipes).toHaveLength(1));
    const requests = fetch.mock.calls.filter(([url]) => String(url).endsWith('/experiments/save')).map(([, init]) => JSON.parse(init.body));
    expect(requests).toHaveLength(2);
    expect(requests[0].idempotency_key).toBe(requests[1].idempotency_key);
  });
});
