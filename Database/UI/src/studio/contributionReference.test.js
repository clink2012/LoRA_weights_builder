import { describe, expect, it } from "vitest";
import { captureContributionReference, contributionForRecord, guidanceForRecord } from "./contributionReference";
const labels = ["BASE", ...Array.from({ length: 19 }, (_, n) => `DOUBLE ${n}`), ...Array.from({ length: 38 }, (_, n) => `SINGLE ${n}`)];
const norms = labels.map((_, index) => index);
const node = { stable_id: "A", profile_version_id: "personal", profile_default_id: "root", loader_export: { architecture_slot_labels: labels } };
const result = { preparation_digest: "digest", node_payloads: [node] };
const job = { job_id: "job", status: "complete", preparation_digest: "digest", entries: [{ stable_id: "A", profile_version_id: "personal" }], metrics: { status: "complete", measurement_basis: "effective_native_parameter_update", outer_model_strength_applied: false, block_weights_applied: false, slot_labels: labels, sources: [{ source_index: 0, block_norms: norms }] } };
const record = { root: { version_id: "root" }, selected: { version_id: "personal", default_id: "root", binding: { slots: labels.map((label) => ({ label })) } } };

describe("Contribution display reference identity", () => {
  it("retains original measurements for edits within the same Default lineage", () => {
    const reference = captureContributionReference(job, result);
    expect(contributionForRecord(reference, "A", { ...record, draft: { values: [99] } }).norms).toEqual(norms);
    expect(contributionForRecord(reference, "A", { ...record, selected: { ...record.selected, version_id: "new-personal" } })).toBeTruthy();
  });
  it.each(["digest", "order", "version", "norm", "labels", "strength", "root"])("rejects a mismatched or invalid %s measurement", (mode) => {
    const changed = structuredClone(job); const prepared = structuredClone(result);
    if (mode === "digest") changed.preparation_digest = "other";
    if (mode === "order") changed.entries[0].stable_id = "B";
    if (mode === "version") changed.entries[0].profile_version_id = "other";
    if (mode === "norm") changed.metrics.sources[0].block_norms[2] = Infinity;
    if (mode === "labels") changed.metrics.slot_labels[2] = "other";
    if (mode === "strength") changed.metrics.outer_model_strength_applied = true;
    if (mode === "root") delete prepared.node_payloads[0].profile_default_id;
    expect(captureContributionReference(changed, prepared)).toBeNull();
  });
  it("hides the reference after Default lineage or slots change", () => {
    const reference = captureContributionReference(job, result);
    expect(contributionForRecord(reference, "A", { ...record, root: { version_id: "changed-source" } })).toBeNull();
    expect(contributionForRecord(reference, "A", { ...record, selected: { ...record.selected, binding: { slots: [{ label: "different" }] } } })).toBeNull();
  });
  it("uses trial guidance only for the measured job, exact version, labels and original norms", () => {
    const reference = captureContributionReference(job, result);
    const guide = labels.map((label, index) => ({ slot_index: index, slot_label: label, state: "review", reason: "Review" }));
    const proposal = { job_id: "job", policy_preview: { contribution_graphs: [{ stable_id: "A", profile_version_id: "personal", original_norms: norms, guidance: guide }] } };
    expect(guidanceForRecord(proposal, reference, "A", record)).toEqual(guide);
    expect(guidanceForRecord({ ...proposal, job_id: "other" }, reference, "A", record)).toBeNull();
    const changed = structuredClone(proposal); changed.policy_preview.contribution_graphs[0].guidance[2].trial_interval = { min: 2, max: 1, basis: "Invalid" };
    expect(guidanceForRecord(changed, reference, "A", record)).toBeNull();
  });
});
