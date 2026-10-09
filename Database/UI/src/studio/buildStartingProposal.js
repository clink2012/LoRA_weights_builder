import { readLoaderExport } from './exportContract';

const TARGET = 'flux1-dev-native-v1';
const active = (job) => ['queued', 'running'].includes(job?.status);
async function request(url, body, signal) {
  const response = await fetch(url, { ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }), signal });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.detail?.reason || data.detail?.excluded_loras?.[0]?.reason_detail || `Preparation failed (${response.status})`);
  return data;
}
function abortError() { return new DOMException('Preparation cancelled', 'AbortError'); }
function wait(signal) {
  return new Promise((resolve, reject) => {
    const cancel = () => { clearTimeout(timer); reject(abortError()); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', cancel); resolve(); }, 750);
    signal.addEventListener('abort', cancel, { once: true });
    if (signal.aborted) cancel();
  });
}

export async function buildStartingProposal(apiBase, ids, { signal, onProgress, force = false }) {
  if (ids.length < 2 || ids.length > 8 || new Set(ids).size !== ids.length) throw new Error('Choose two to eight distinct LoRAs for a starting proposal.');
  onProgress('Checking original Defaults…');
  const roots = [];
  for (const id of ids) {
    const root = await request(`${apiBase}/profile-versions/${encodeURIComponent(id)}/defaults`, {}, signal);
    if (root.kind !== 'default' || root.stable_id !== id || root.default_id !== root.version_id) throw new Error('The original profile response does not match this stack.');
    roots.push(root);
  }
  const entries = roots.map((root) => ({ stable_id: root.stable_id, profile_version_id: root.version_id }));
  const source = await request(`${apiBase}/lora/prepare-blocks`, { stable_ids: ids, target_contract_id: TARGET, profile_version_ids: Object.fromEntries(entries.map((e) => [e.stable_id, e.profile_version_id])) }, signal);
  if (!source.compatible || !/^[a-f0-9]{64}$/.test(source.preparation_digest || '') || source.node_payloads?.length !== ids.length || source.node_payloads.some((node, i) => node.stable_id !== ids[i] || node.profile_version_id !== entries[i].profile_version_id || !readLoaderExport(node).ready)) throw new Error('The complete stack could not be prepared against current sources.');
  const body = { entries, target_contract_id: TARGET, expected_preparation_digest: source.preparation_digest };
  let job, ownedJob;
  const cancelOwned = () => { if (ownedJob && active(job)) request(`${apiBase}/analysis-jobs/${encodeURIComponent(ownedJob)}/cancel`, {}).catch(() => {}); };
  signal.addEventListener('abort', cancelOwned);
  try {
    onProgress('Checking saved measurements…');
    const resolved = await request(`${apiBase}/analysis-jobs/resolve`, body, signal);
    if (resolved.status === 'reused') job = resolved.job;
    else if (resolved.status === 'not_found' && resolved.job === null) {
      onProgress('Reading LoRA tensors and measuring overlap…');
      // Receive the created job ID even if selection changes during this POST,
      // so the obsolete worker can be cancelled instead of orphaned.
      job = await request(`${apiBase}/analysis-jobs`, body);
      ownedJob = job.job_id;
    } else throw new Error('Saved measurement lookup returned an invalid response.');
    function validate() {
      if (signal.aborted) { cancelOwned(); throw abortError(); }
      if (!job?.job_id || job.target_contract_id !== TARGET || job.preparation_digest !== source.preparation_digest || !Array.isArray(job.entries) || job.entries.length !== entries.length || job.entries.some((e, i) => e.stable_id !== entries[i].stable_id || e.profile_version_id !== entries[i].profile_version_id)) throw new Error('The measurements do not match this stack.');
    }
    validate();
    while (active(job)) {
      await wait(signal);
      job = await request(`${apiBase}/analysis-jobs/${encodeURIComponent(job.job_id)}`, undefined, signal);
      validate();
    }
    if (job.status !== 'complete') throw new Error(job.reason || `Measurement ${job.status}; no proposal was produced.`);
    onProgress('Computing role-aware block values…');
    const data = await request(`${apiBase}/experiments/prepare`, { job_id: job.job_id, priorities: {}, expected_preparation_digest: source.preparation_digest, force_recompute: force }, signal);
    const proposal = data.starting_proposal;
    if (!data.compatible || !/^[a-f0-9]{64}$/.test(data.preparation_digest || '') || !/^[a-f0-9]{64}$/.test(proposal?.proposal_digest || '') || typeof proposal?.computed_baseline?.reused !== 'boolean' || proposal?.job_id !== job.job_id || proposal.input_preparation_digest !== source.preparation_digest || proposal.policy_preview?.policy_version !== 'managed_role_start_v1' || proposal.policy_preview.calibrated !== false || !Array.isArray(proposal.policy_preview.entries) || proposal.policy_preview.entries.length !== ids.length || !Array.isArray(data.source_profiles) || data.source_profiles.length !== ids.length || !Array.isArray(data.node_payloads) || data.node_payloads.length !== ids.length || data.node_payloads.some((node, i) => {
      const entry = proposal.policy_preview.entries[i], parent = data.source_profiles[i];
      return node.stable_id !== ids[i] || entry.stable_id !== ids[i] || entry.profile_version_id !== entries[i].profile_version_id || parent.stable_id !== ids[i] || parent.version_id !== entries[i].profile_version_id || parent.default_id !== parent.version_id || parent.kind !== 'default' || JSON.stringify(parent.values) !== JSON.stringify(roots[i].values) || JSON.stringify(entry.before_values) !== JSON.stringify(roots[i].values) || !readLoaderExport(node).ready || JSON.stringify(node.loader_export.architecture_slot_values) !== JSON.stringify(entry.values);
    })) throw new Error('The proposed graph and loader values do not match the reviewed stack.');
    return { ...data, measurement_job: job };
  } finally {
    cancelOwned();
    signal.removeEventListener('abort', cancelOwned);
  }
}
