export function captureContributionReference(job, result) {
  const metrics = job?.metrics;
  const nodes = result?.node_payloads;
  if (job?.status !== "complete" || job.preparation_digest !== result?.preparation_digest || metrics?.status !== "complete" || metrics.measurement_basis !== "effective_native_parameter_update" || metrics.outer_model_strength_applied !== false || metrics.block_weights_applied !== false || !Array.isArray(metrics.slot_labels) || metrics.slot_labels.length !== 58 || !Array.isArray(nodes) || !Array.isArray(job.entries) || !Array.isArray(metrics.sources) || nodes.length !== job.entries.length || nodes.length !== metrics.sources.length) return null;
  const sources = {};
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index]; const entry = job.entries[index]; const source = metrics.sources[index];
    if (node.stable_id !== entry.stable_id || node.profile_version_id !== entry.profile_version_id || typeof node.profile_default_id !== "string" || source.source_index !== index || !Array.isArray(source.block_norms) || source.block_norms.length !== 58 || !source.block_norms.every((value) => Number.isFinite(value) && value >= 0) || JSON.stringify(node.loader_export?.architecture_slot_labels) !== JSON.stringify(metrics.slot_labels)) return null;
    sources[node.stable_id] = { defaultId: node.profile_default_id, norms: [...source.block_norms], labels: [...metrics.slot_labels], jobId: job.job_id };
  }
  return { jobId: job.job_id, sources };
}

export function contributionForRecord(reference, id, record) {
  const source = reference?.sources?.[id];
  if (!source || !record?.selected || !record.root || source.defaultId !== record.root.version_id || record.selected.default_id !== source.defaultId || JSON.stringify(record.selected.binding.slots.map((slot) => slot.label)) !== JSON.stringify(source.labels)) return null;
  return source;
}

export function guidanceForRecord(proposal, reference, id, record) {
  if (!proposal || proposal.job_id !== reference?.jobId) return null;
  const graph = proposal.policy_preview?.contribution_graphs?.find((entry) => entry.stable_id === id && entry.profile_version_id === record?.selected?.version_id);
  const source = contributionForRecord(reference, id, record);
  if (!source || !graph || JSON.stringify(graph.original_norms) !== JSON.stringify(source.norms) || !Array.isArray(graph.guidance) || graph.guidance.length !== 58 || graph.guidance.some((entry, index) => entry.slot_index !== index || entry.slot_label !== source.labels[index] || !["inactive", "protected", "suggested", "review"].includes(entry.state) || typeof entry.reason !== "string" || (entry.trial_interval && (!Number.isFinite(entry.trial_interval.min) || !Number.isFinite(entry.trial_interval.max) || entry.trial_interval.min > entry.trial_interval.max || typeof entry.trial_interval.basis !== "string")))) return null;
  return graph.guidance;
}
