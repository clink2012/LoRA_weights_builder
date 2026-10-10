import { useCallback, useState } from "react";
import { readLoaderExport } from "./exportContract";
import "./Studio.css";
import { useProfileVariants } from "./useProfileVariants";
import VersionPanel from "./VersionPanel";
import CompositionPanel from "./CompositionPanel";
import LoraThumbnail from "./LoraThumbnail";
import MeasurementPanel from "./MeasurementPanel";
import ExperimentPanel from "./ExperimentPanel";
import RenderTrialPanel from "./RenderTrialPanel";
import BlockChart from "./BlockChart";
import MeasuredBlockChart from "./MeasuredBlockChart";
import { captureContributionReference, contributionForRecord, guidanceForRecord } from "./contributionReference";
import LibraryScanStatus from "./LibraryScanStatus";
import H3InspectionPanel from "./H3InspectionPanel";

const COLOURS = ["#bc9cff", "#4bd6e4", "#f6b567", "#ef8cae", "#9cda95"];
const nameOf = (item) => (item?.filename || item?.stable_id || "LoRA").replace(/\.safetensors$/i, "");

function LibraryItem({ item, picked, onToggle, onInspect, disabled, apiBase }) {
  const inspectOnly = item.base_model_code === "MH3";
  return <button className={`studio-library-item ${picked ? "is-picked" : ""}`} onClick={() => inspectOnly ? onInspect(item) : onToggle(item.stable_id)} aria-label={inspectOnly ? `Inspect H3 ${nameOf(item)}` : undefined} aria-pressed={inspectOnly ? undefined : picked} disabled={disabled}>
    <LoraThumbnail apiBase={apiBase} stableId={item.stable_id} name={nameOf(item)} />
    <span className="studio-library-text"><strong>{nameOf(item)}</strong><small>{item.role || item.category_code || "Role unknown"} · {item.base_model_code}</small><small>{item.stable_id}</small>{item.compatibility && item.compatibility.status !== "eligible" && <small className="studio-presence-note">{item.compatibility.status === "unknown" ? "Not established" : "Excluded"}: {item.compatibility.reason}</small>}{item.presence && item.presence !== "current" && <small className="studio-presence-note">{item.presence === "missing" ? "Missing file · history retained" : item.presence === "out_of_scope" ? "Outside current library" : "Not checked yet"}</small>}</span>
    <span className="studio-pick" aria-hidden="true">{inspectOnly ? "Inspect" : picked ? "✓" : "+"}</span>
  </button>;
}

function Comparison({ items, getDisplayContract }) {
  const [hidden, setHidden] = useState([]);
  const series = items.map((item, index) => ({ item, colour: COLOURS[index % COLOURS.length], contract: getDisplayContract(item.stable_id) })).filter((entry) => entry.contract.ready);
  if (!series.length) return null;
  const reference = series[0].contract;
  const aligned = series.filter(({ contract }) => contract.slots.length === reference.slots.length && contract.slots.every((slot, index) => slot.label === reference.slots[index].label));
  const max = Math.max(1, ...aligned.flatMap(({ contract }) => contract.slots.map((slot) => Math.abs(slot.value))));
  return <section className="studio-panel studio-comparison"><div className="studio-section-heading"><div><span className="studio-eyebrow">See the whole composition</span><h2>Block comparison</h2></div><span className="studio-tag">Signed values</span></div>
    <svg viewBox="0 0 800 180" role="img" aria-label="Overlay of aligned individual block values">
      <line x1="10" y1="90" x2="790" y2="90" className="studio-zero" />
      {aligned.filter(({ item }) => !hidden.includes(item.stable_id)).map(({ item, contract, colour }) => <polyline key={item.stable_id} fill="none" stroke={colour} strokeWidth="2.5" points={contract.slots.map((slot, index) => `${10 + index / Math.max(1, contract.slots.length - 1) * 780},${90 - slot.value / max * 72}`).join(" ")} />)}
      <text x="12" y="20">+{max}</text><text x="12" y="174">−{max}</text>
    </svg>
    <div className="studio-legend">{aligned.map(({ item, colour }) => <button key={item.stable_id} aria-pressed={!hidden.includes(item.stable_id)} onClick={() => setHidden((previous) => previous.includes(item.stable_id) ? previous.filter((id) => id !== item.stable_id) : [...previous, item.stable_id])}><i style={{ background: colour }} />{nameOf(item)}</button>)}</div>
    <p className="studio-help">Overlap shows shared block positions. It does not measure image conflict or guarantee a better image.</p>
    {aligned.length < series.length && <p className="studio-help">Different slot mappings are excluded from this overlay.</p>}
  </section>;
}

function LoaderCard({ item, payload, index, loading }) {
  const [copyState, setCopyState] = useState("");
  const contract = readLoaderExport(payload);
  async function copy() {
    try { await navigator.clipboard.writeText(contract.csv); setCopyState("Copied"); }
    catch { setCopyState("Copy failed — select the full text below to copy manually."); }
  }
  return <article className="studio-loader-card"><div className="studio-section-heading"><div><span className="studio-eyebrow">Loader {index + 1} · {payload?.profile_name || "Server result"}</span><h3>{nameOf(item)}</h3></div><span className={`studio-tag ${contract.ready ? "is-ready" : ""}`}>{contract.ready ? `${contract.loader_slot_count} loader slots` : "Export unavailable"}</span></div>
    {contract.ready ? <><p className="studio-help">{contract.target_contract_label || "FLUX.1 dev · standard 19 double / 38 single"} · {contract.contract_id}</p><p className="studio-help">{contract.recommendation_basis === "manual_variant_unvalidated" ? "Personal variant: these are your saved values, not an image-tested recommendation." : contract.recommendation_basis === "structural_baseline_unvalidated" ? "Structural baseline only: present blocks start enabled. This is not yet a balanced recommendation or an image-tested recipe." : "Heuristic result: image quality has not been validated."} Confirm your workflow uses the named target before copying.</p><textarea readOnly value={contract.csv} aria-label={`Full block values for ${nameOf(item)}`} rows={4} /><div className="studio-loader-footer"><span>Model strength {payload.strength_model ?? "—"} · CLIP {payload.strength_clip ?? "not applicable"} · Inverse off</span><button className="studio-primary" disabled={loading} onClick={copy}>Copy full vector</button></div><span role="status">{copyState}</span></> : <p className="studio-help">{contract.reason}</p>}
  </article>;
}

export default function Studio({ apiBase, libraryFamily = "FLX", currentRecipe, onRecipeSaved, versionIds, draftProfiles, onVersionChange, onDraftChange, onRestoreComposition, onInvalidatePrepared, catalog, selectedItems, selectedIds, computedById, result, error, loading, preparationProgress, loadedValues, catalogLoading, libraryRefreshing, catalogueStatus, scan, compatibilitySummary, libraryPresence, catalogError, search, onSearch, onSearchSubmit, onToggle, onRemove, onClear, onCalculate, page, pages, onPage, showAll, onShowAll }) {
  const [h3Inspection, setH3Inspection] = useState(null);
  const [focusedId, setFocusedId] = useState(null);
  const [selectedSlot, setSelectedSlot] = useState(0);
  const [recipeBusy, setRecipeBusy] = useState(false);
  const [experimentBusy, setExperimentBusy] = useState(false);
  const [analysisJob, setAnalysisJob] = useState(null);
  const [contributionReference, setContributionReference] = useState(null);
  const [graphProposal, setGraphProposal] = useState(null);
  const [seenProposalResult, setSeenProposalResult] = useState(null);
  if (result?.starting_proposal && result !== seenProposalResult) {
    setSeenProposalResult(result);
    const reference = captureContributionReference(result.measurement_job, result.source_preparation);
    if (reference) { setContributionReference(reference); setGraphProposal(result.starting_proposal); }
  }
  const handleMeasurement = useCallback((job) => {
    setAnalysisJob(job);
    const captured = libraryRefreshing ? null : captureContributionReference(job, result);
    if (captured) setContributionReference((previous) => previous?.jobId === captured.jobId ? previous : captured);
  }, [result, libraryRefreshing]);
  const [seenLibraryRefresh, setSeenLibraryRefresh] = useState(libraryRefreshing);
  if (seenLibraryRefresh !== libraryRefreshing) {
    setSeenLibraryRefresh(libraryRefreshing);
    if (libraryRefreshing) { setContributionReference(null); setGraphProposal(null); }
  }
  const [workspaceView, setWorkspaceView] = useState("build");
  const selected = selectedItems.find((item) => item.stable_id === focusedId) || selectedItems[0];
  const variants = useProfileVariants(apiBase, versionIds, onVersionChange, onDraftChange, result);
  const dirty = selectedIds.some((id) => draftProfiles[id]);
  const baseProfileBusy = libraryRefreshing || recipeBusy || Object.values(variants.records).some((record) => record.busy);
  const profileOperationBusy = experimentBusy || baseProfileBusy;
  const operationBusy = loading || profileOperationBusy;
  const h3Library = libraryFamily === "MH3" && !selectedItems.length;
  function getDisplayContract(id) {
    const record = variants.records[id];
    const snapshot = record?.draft || record?.computed || record?.selected;
    if (snapshot) return { ready: true, slots: snapshot.binding ? snapshot.binding.slots.map((slot, index) => ({ ...slot, value: snapshot.values[index] })) : record.selected.binding.slots.map((slot, index) => ({ ...slot, value: snapshot.values[index] })) };
    return readLoaderExport(computedById.get(id));
  }
  const contract = getDisplayContract(selected?.stable_id);
  const selectedRecord = variants.records[selected?.stable_id];
  const measured = result?.compatible === false ? null : contributionForRecord(contributionReference, selected?.stable_id, selectedRecord);
  const measuredGuidance = guidanceForRecord(graphProposal, contributionReference, selected?.stable_id, selectedRecord);
  const suggestion = measuredGuidance && graphProposal?.policy_preview?.entries.find((entry) => entry.stable_id === selected?.stable_id && entry.profile_version_id === selectedRecord.selected.version_id)?.values;
  const suggestedValues = Array.isArray(suggestion) && suggestion.length === 58 && suggestion.every(Number.isFinite) ? suggestion : null;
  const slot = contract.ready ? contract.slots[selectedSlot] || contract.slots[0] : null;
  function changeWorkspace(event, view) {
    const next = event.key === "Home" ? "build" : event.key === "End" ? "compare" : ["ArrowLeft", "ArrowRight"].includes(event.key) ? view === "build" ? "compare" : "build" : null;
    if (!next) return;
    event.preventDefault(); setWorkspaceView(next); document.getElementById(`studio-tab-${next}`)?.focus();
  }
  return <div className="studio-workspace"><aside className="studio-panel studio-library"><div className="studio-section-heading"><div><span className="studio-eyebrow">Your collection</span><h2>LoRA library</h2></div><span className="studio-tag">{catalog.length}</span></div>
    <LibraryScanStatus apiBase={apiBase} scan={scan} />
    <form className="studio-library-search" onSubmit={onSearchSubmit}><label className="studio-search">Find a LoRA<input disabled={libraryRefreshing} value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search your library…" /></label><button type="submit" disabled={catalogLoading}>Search</button></form>
    <p className="studio-catalogue-hint">Family and role labels are folder hints, not verified architecture or export support. Preparing a stack checks the actual files.</p>
    {catalogueStatus === "refreshed" && <p className="studio-catalogue-hint">Current files reflects the last successful refresh. Refresh again after adding, moving or removing LoRA files.</p>}
    {catalogueStatus === "not_refreshed" && <p className="studio-help">Choose “Refresh library” to check current files. Until then, older records are available under All catalogue entries.</p>}
    {libraryPresence === "missing" && <p className="studio-help">These files were missing at the last refresh. Their catalogue and saved history remain; exports require the source file to be available again.</p>}
    {selectedIds.length > 0 && libraryPresence === "current" && <><label className="studio-check"><input type="checkbox" checked={showAll} onChange={(event) => onShowAll(event.target.checked)} />Show excluded and unverified files</label><p className="studio-catalogue-hint">Checked against the first selected LoRA and the pinned FLUX.1 loader target. Structural eligibility does not guarantee a good image.</p>{compatibilitySummary?.counts && <p className="studio-catalogue-hint">{compatibilitySummary.counts.eligible} eligible · {compatibilitySummary.counts.excluded} excluded · {compatibilitySummary.counts.unknown} unverified</p>}</>}
    {selectedIds.length > 0 && libraryPresence !== "current" && <p className="studio-catalogue-hint">Compatibility filtering applies to Current files. This view retains catalogue history.</p>}
    {catalogLoading && <p role="status">Loading your library…</p>}{catalogError && <p role="alert">{catalogError}</p>}
    {!catalogLoading && !catalog.length && catalogueStatus !== "not_refreshed" && <p className="studio-help">No matching LoRAs. Try changing the library filters.</p>}
    <div className="studio-library-list">{catalog.map((item) => <LibraryItem key={item.stable_id} apiBase={apiBase} item={item} picked={selectedIds.includes(item.stable_id)} onToggle={onToggle} onInspect={setH3Inspection} disabled={catalogLoading || profileOperationBusy || Boolean(draftProfiles[item.stable_id]) || (item.base_model_code !== "MH3" && !selectedIds.includes(item.stable_id) && selectedIds.length > 0 && libraryPresence === "current" && item.compatibility?.status !== "eligible")} />)}</div>
    <div className="studio-pagination"><button onClick={() => onPage(page - 1)} disabled={page === 0 || catalogLoading} aria-label="Previous library page">←</button><span>{page + 1} / {pages}</span><button onClick={() => onPage(page + 1)} disabled={page + 1 >= pages || catalogLoading} aria-label="Next library page">→</button></div>
  </aside><div className="studio-canvas"><section className="studio-panel studio-stack" aria-label="Selected stack"><div className="studio-section-heading"><div><span className="studio-eyebrow">Build your composition</span><h2>Selected stack <span>{selectedItems.length}</span></h2></div><button className="studio-text-button" onClick={onClear} disabled={!selectedItems.length || operationBusy || dirty}>Clear stack</button></div>
    {selectedItems.length ? <div className="studio-stack-items">{selectedItems.map((item, index) => <div key={item.stable_id} className={`studio-stack-item ${selected?.stable_id === item.stable_id ? "is-active" : ""}`}><button onClick={() => { setFocusedId(item.stable_id); setSelectedSlot(0); }}><i style={{ background: COLOURS[index % COLOURS.length] }} /><span><small>Loader {index + 1} · {item.role || "Role unknown"}</small><strong>{nameOf(item)}</strong></span></button><button disabled={operationBusy || Boolean(draftProfiles[item.stable_id])} onClick={() => onRemove(item.stable_id)} aria-label={`Remove ${nameOf(item)}`}>×</button></div>)}</div> : <p className="studio-help">Choose LoRAs from the library. Each gets its own complete block vector in the order you add it.</p>}
    <div className="studio-action-row"><p>Target: <strong>{h3Library ? "MiniMax H3 · adapter inspection" : "FLUX.1 dev · Inspire block loader"}</strong><span>{h3Library ? "Choose an H3 adapter to inspect. Block-weight preparation and export are not enabled yet." : loadedValues ? "Loaded values will be checked without recalculation" : selectedIds.length >= 2 ? "Build a role-aware starting proposal from original Defaults" : "Select another LoRA to build a stack proposal"}</span></p><button className="studio-primary" onClick={onCalculate} disabled={!selectedItems.length || operationBusy || dirty}>{loading ? "Preparing…" : "Prepare block values"}</button>{selectedIds.length >= 2 && <button disabled={operationBusy || dirty} onClick={() => onCalculate({ fresh: true })}>Build a fresh proposal</button>}{loading && <button onClick={onInvalidatePrepared}>Cancel preparation</button>}</div>
    {loading && preparationProgress && <p role="status">{preparationProgress}</p>}
    {dirty && <p className="studio-help">Pending personal drafts: {selectedItems.filter((item) => draftProfiles[item.stable_id]).map(nameOf).join(', ')}. Editing a starting proposal keeps every proposed vector as a draft. Save or discard each draft before preparing or copying the stack.</p>}
    {result?.starting_proposal && <div className="studio-proposal-summary"><p role="status">Role-aware starting values {result.starting_proposal.computed_baseline.reused ? "reused after fresh validation" : "computed"}. Graphs and Copy use this proposal. Save it as a recipe if you want to keep it as a personal option.</p><p className="studio-help">Declared role levels followed by measured overlap checks; no block is assumed to control a face, coat or shoe. These values are an experimental starting point.</p>{result.starting_proposal.policy_preview.role_rules.map((rule) => <p key={rule.stable_id}>{nameOf(selectedItems.find((item) => item.stable_id === rule.stable_id))}: {rule.normalized_role} · starting multiplier ×{rule.starting_factor}</p>)}{!result.starting_proposal.policy_preview.changes.length && <p className="studio-help">No change qualifies for this stack. Check unknown roles and the numerical guard details before experimenting.</p>}</div>}
    {dirty && <p className="studio-help">Personal draft changes are visible below. Save or discard them before preparing a new export.</p>}
    {error && <p role="alert" className="studio-alert">{error}</p>}
    {result && <details className="studio-evidence"><summary>Calculation details</summary><p>{result.compatible === false ? "Needs attention" : "Header and loader checks passed"} · {result.validated_base_model} · {result.validated_layout}</p>{[...(result.reasons || []), ...(result.warnings || [])].map((text, index) => <p key={index}>{typeof text === "string" ? text : text?.reason_detail || text?.reason || "A selected LoRA could not be prepared."}</p>)}</details>}
  </section>
  {h3Inspection && <H3InspectionPanel key={h3Inspection.stable_id} apiBase={apiBase} item={h3Inspection} onClose={() => setH3Inspection(null)} />}
  <div className="studio-workspace-tabs" role="tablist" aria-label="Studio workspace">{[["build", "Build"], ["compare", "Compare & experiment"]].map(([view, label]) => <button key={view} type="button" role="tab" id={`studio-tab-${view}`} aria-controls={`studio-panel-${view}`} aria-selected={workspaceView === view} tabIndex={workspaceView === view ? 0 : -1} onClick={() => setWorkspaceView(view)} onKeyDown={(event) => changeWorkspace(event, view)}>{label}</button>)}</div>
  <div className="studio-workspace-page" id="studio-panel-build" role="tabpanel" aria-labelledby="studio-tab-build" hidden={workspaceView !== "build"}>
  {h3Library ? <section className="studio-panel studio-editor"><div className="studio-editor-empty"><span className="studio-orbit" aria-hidden="true">▥</span><h3>Inspect an H3 adapter</h3><p>Choose Inspect in the library to see observed main-block, refiner and other coverage. Proposed H3 multipliers and ComfyUI export are not available yet.</p></div></section> : <section className="studio-panel studio-editor"><div className="studio-section-heading"><div><span className="studio-eyebrow">Individual block weights</span><h2>{selected ? nameOf(selected) : "Make room for every LoRA"}</h2></div><span className="studio-tag">{variants.records[selected?.stable_id]?.draft ? "Personal draft" : variants.records[selected?.stable_id]?.computed?.name || variants.records[selected?.stable_id]?.selected?.name || "Server result"}</span></div>
    {contract.ready ? <>{measured ? <MeasuredBlockChart record={selectedRecord} reference={measured} guidance={measuredGuidance} suggestedValues={suggestedValues} selectedSlot={selectedSlot} onSelect={setSelectedSlot} onEditValue={(index, value) => variants.editValue(selected.stable_id, index, value)} onEdit={(update) => variants.edit(selected.stable_id, update)} disabled={operationBusy} /> : <><BlockChart contract={contract} selectedSlot={selectedSlot} onSelect={setSelectedSlot} onEditValue={(index, value) => selectedRecord?.selected ? variants.editValue(selected.stable_id, index, value) : variants.open(selected.stable_id, { index, value })} originalValues={selectedRecord?.root?.values} savedValues={selectedRecord?.selected?.values} disabled={operationBusy} /><p className="studio-help">Press and hold to draw across bars, or edit a number above, to create a personal draft. The first edit captures your Default and opens history. The dashed line shows captured Default multipliers; measuring in Compare & experiment adds the original update sizes.</p>{selectedRecord?.root && <div className="studio-measured-actions"><button disabled={operationBusy} onClick={() => variants.editValue(selected.stable_id, selectedSlot, selectedRecord.root.values[selectedSlot])}>Reset {slot.label} to Default</button><button disabled={operationBusy} onClick={() => variants.edit(selected.stable_id, { values: [...selectedRecord.root.values], ab: {} })}>Reset all blocks to Default</button></div>}</>}<div className="studio-inspector"><div><span className="studio-eyebrow">Selected slot</span><strong>{slot.label}</strong></div><label>Exact value<input type="number" readOnly value={slot.value} /></label><p>Default stays unchanged. Save your edits as a personal revision. Loader order can differ for sparse adapters.</p></div></> : <div className="studio-editor-empty"><span className="studio-orbit" aria-hidden="true">▥</span><h3>{selected ? "Your block workspace" : "Start with your first LoRA"}</h3><p>{selected ? result ? contract.reason : "Prepare the stack to check its loader mapping and inspect individual blocks." : "Select a person, clothing, style or another LoRA from your library."}</p></div>}
    {selected && <VersionPanel id={selected.stable_id} record={variants.records[selected.stable_id]} slotIndex={selectedSlot} actions={variants} loading={operationBusy} />}
  </section>}
  {result && <section className="studio-exports" aria-label="Full loader vectors"><div className="studio-section-heading"><div><span className="studio-eyebrow">Take it into ComfyUI</span><h2>One full vector per loader</h2></div></div>{selectedItems.map((item, index) => <LoaderCard key={item.stable_id} item={item} index={index} payload={computedById.get(item.stable_id)} loading={operationBusy} />)}</section>}
  </div>
  <div className="studio-workspace-page" id="studio-panel-compare" role="tabpanel" aria-labelledby="studio-tab-compare" hidden={workspaceView !== "compare"}>
  <Comparison items={selectedItems} getDisplayContract={getDisplayContract} />
  <MeasurementPanel apiBase={apiBase} selectedItems={selectedItems} versionIds={versionIds} result={result} dirty={dirty} loading={loading || baseProfileBusy} experimentBusy={experimentBusy} onJobChange={handleMeasurement} onInvalidatePrepared={onInvalidatePrepared} />
  <ExperimentPanel apiBase={apiBase} job={analysisJob} selectedItems={selectedItems} versionIds={versionIds} result={result} dirty={dirty} loading={loading || baseProfileBusy} currentRecipe={currentRecipe} onBusyChange={setExperimentBusy} onRestore={onRestoreComposition} onInvalidatePrepared={onInvalidatePrepared} onGuidance={setGraphProposal} selectedId={selected?.stable_id} selectedSlot={selectedSlot} />
  <RenderTrialPanel apiBase={apiBase} currentRecipe={currentRecipe} selectedIds={selectedIds} versionIds={versionIds} dirty={dirty} loading={operationBusy} />
  </div>
  <CompositionPanel apiBase={apiBase} selectedIds={selectedIds} versionIds={versionIds} result={result} dirty={dirty} loading={operationBusy} onBusyChange={setRecipeBusy} currentRecipe={currentRecipe} onSaved={onRecipeSaved} onRestore={onRestoreComposition} onVersionChange={onVersionChange} onInvalidatePrepared={onInvalidatePrepared} />
  </div></div>;
}
