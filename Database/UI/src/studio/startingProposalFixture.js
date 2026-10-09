// Synthetic fixtures only; no model files, user database or rendered evidence.
export const labels = ['BASE', ...Array.from({ length: 19 }, (_, i) => `DOUBLE ${i}`), ...Array.from({ length: 38 }, (_, i) => `SINGLE ${i}`)];
export const target = 'flux1-dev-native-v1';
export const digest = 'a'.repeat(64);
export function root(id, role = id === 'A' ? 'character' : 'clothing') {
  return { version_id: `default-${id}`, default_id: `default-${id}`, stable_id: id, kind: 'default', name: 'Default', sequence: 1,
    values: labels.map(() => 1), settings: { role, strength_model: 1, strength_clip: 0, affect_clip: false }, ab: {}, binding: { slots: labels.map((label) => ({ label, group: label.split(' ')[0].toLowerCase() })) } };
}
export function node(profile) {
  return { stable_id: profile.stable_id, filename: `${profile.stable_id}.safetensors`, profile_version_id: profile.version_id, profile_default_id: profile.default_id, ...profile.settings, profile_name: profile.name,
    loader_export: { status: 'ready', adapter_id: 'inspire_flux1_v1', architecture_slot_labels: labels, architecture_slot_values: profile.values, architecture_slot_count: 58, slot_labels: labels, slot_values: profile.values, loader_slot_count: 58, numeric_csv: profile.values.join(',') } };
}
export function source(ids = ['A', 'B']) {
  return { compatible: true, target_contract_id: target, preparation_digest: digest, node_payloads: ids.map((id) => node(root(id))) };
}
export function job(status = 'complete') {
  return { job_id: 'measured', status, target_contract_id: target, preparation_digest: digest, entries: ['A', 'B'].map((id) => ({ stable_id: id, profile_version_id: `default-${id}` })),
    metrics: { status: 'complete', measurement_basis: 'effective_native_parameter_update', outer_model_strength_applied: false, block_weights_applied: false, slot_labels: labels,
      sources: [0, 1].map((source_index) => ({ source_index, block_norms: labels.map(() => 1), block_squared_norms: labels.map(() => 1), total_squared_norm: 58 })), pairs: [] } };
}
export function proposal() {
  const roots = ['A', 'B'].map((id) => root(id));
  const entries = roots.map((r, i) => ({ stable_id: r.stable_id, profile_version_id: r.version_id, before_values: r.values, values: r.values.map((value, slot) => slot ? [.9, .65][i] : value) }));
  const graphs = entries.map((e) => ({ stable_id: e.stable_id, profile_version_id: e.profile_version_id, original_norms: labels.map(() => 1), guidance: labels.map((slot_label, slot_index) => ({ slot_label, slot_index, state: slot_index ? 'suggested' : 'protected', reason: 'Declared role starting factor', trial_interval: null })) }));
  return { compatible: true, preparation_digest: 'b'.repeat(64), source_preparation: source(), source_profiles: roots,
    node_payloads: entries.map((e, i) => ({ ...node({ ...roots[i], values: e.values, name: 'Role-aware starting proposal' }), profile_version_id: null })),
    starting_proposal: { job_id: 'measured', input_preparation_digest: digest, proposal_digest: 'c'.repeat(64), computed_baseline: { reused: false }, policy_preview: { policy_version: 'managed_role_start_v1', calibrated: false, entries, contribution_graphs: graphs, changes: [{ stable_id: 'A', slot_index: 1, value: .9 }], role_rules: roots.map((r, i) => ({ stable_id: r.stable_id, normalized_role: r.settings.role, starting_factor: [.9, .65][i] })) } } };
}
