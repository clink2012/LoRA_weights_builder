import { useState } from "react";
import { readLoaderExport } from "./exportContract";
import "./Studio.css";
import { useProfileVariants } from "./useProfileVariants";
import VersionPanel from "./VersionPanel";
import CompositionPanel from "./CompositionPanel";
import LoraThumbnail from "./LoraThumbnail";
import MeasurementPanel from "./MeasurementPanel";
import ExperimentPanel from "./ExperimentPanel";

const COLOURS = ["#bc9cff", "#4bd6e4", "#f6b567", "#ef8cae", "#9cda95"];
const nameOf = (item) => (item?.filename || item?.stable_id || "LoRA").replace(/\.safetensors$/i, "");

function LibraryItem({ item, picked, onToggle, disabled, apiBase }) {
  return <button className={`studio-library-item ${picked ? "is-picked" : ""}`} onClick={() => onToggle(item.stable_id)} aria-pressed={picked} disabled={disabled}>
    <LoraThumbnail apiBase={apiBase} stableId={item.stable_id} name={nameOf(item)} />
    <span className="studio-library-text"><strong>{nameOf(item)}</strong><small>{item.role || item.category_code || "Role unknown"} · {item.base_model_code}</small><small>{item.stable_id}</small>{item.presence && item.presence !== "current" && <small className="studio-presence-note">{item.presence === "missing" ? "Missing file · history retained" : item.presence === "out_of_scope" ? "Outside current library" : "Not checked yet"}</small>}</span>
    <span className="studio-pick" aria-hidden="true">{picked ? "✓" : "+"}</span>
  </button>;
}

function BlockChart({ contract, selectedSlot, onSelect }) {
  const groups = [...new Set(contract.slots.map((slot) => slot.group))];
  const extent = Math.max(1, ...contract.slots.map((slot) => Math.abs(slot.value)));
  return <div className="studio-block-groups">{groups.map((group) => <section key={group} className="studio-block-group">
    <div className="studio-group-heading"><strong>{group}</strong><span>{contract.slots.filter((slot) => slot.group === group).length} slots</span></div>
    <div className="studio-bars">{contract.slots.map((slot, index) => slot.group === group && <button key={index} className={`studio-bar ${selectedSlot === index ? "is-current" : ""} ${slot.value < 0 ? "is-negative" : ""}`} aria-label={`${slot.label}: ${slot.value}`} aria-pressed={selectedSlot === index} onClick={() => onSelect(index)} title={`${slot.label}: ${slot.value}`}>
      <span className="studio-bar-track"><i style={{ height: `${Math.abs(slot.value) / extent * 100}%` }} /></span><span>{slot.label === "BASE" ? "BASE" : slot.label.split(" ").at(-1)}</span>
    </button>)}</div>
  </section>)}</div>;
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

export default function Studio({ apiBase, currentRecipe, onRecipeSaved, versionIds, draftProfiles, onVersionChange, onDraftChange, onRestoreComposition, onInvalidatePrepared, catalog, selectedItems, selectedIds, computedById, result, error, loading, catalogLoading, libraryRefreshing, catalogueStatus, libraryPresence, catalogError, search, onSearch, onSearchSubmit, onToggle, onRemove, onClear, onCalculate, page, pages, onPage, showAll, onShowAll, hiddenCount }) {
  const [focusedId, setFocusedId] = useState(null);
  const [selectedSlot, setSelectedSlot] = useState(0);
  const [recipeBusy, setRecipeBusy] = useState(false);
  const [experimentBusy, setExperimentBusy] = useState(false);
  const [analysisJob, setAnalysisJob] = useState(null);
  const selected = selectedItems.find((item) => item.stable_id === focusedId) || selectedItems[0];
  const variants = useProfileVariants(apiBase, versionIds, onVersionChange, onDraftChange);
  const dirty = selectedIds.some((id) => draftProfiles[id]);
  const baseProfileBusy = libraryRefreshing || recipeBusy || Object.values(variants.records).some((record) => record.busy);
  const profileOperationBusy = experimentBusy || baseProfileBusy;
  const operationBusy = loading || profileOperationBusy;
  function getDisplayContract(id) {
    const record = variants.records[id];
    const snapshot = record?.draft || record?.selected;
    if (snapshot) return { ready: true, slots: snapshot.binding ? snapshot.binding.slots.map((slot, index) => ({ ...slot, value: snapshot.values[index] })) : record.selected.binding.slots.map((slot, index) => ({ ...slot, value: snapshot.values[index] })) };
    return readLoaderExport(computedById.get(id));
  }
  const contract = getDisplayContract(selected?.stable_id);
  const slot = contract.ready ? contract.slots[selectedSlot] || contract.slots[0] : null;
  return <div className="studio-workspace"><aside className="studio-panel studio-library"><div className="studio-section-heading"><div><span className="studio-eyebrow">Your collection</span><h2>LoRA library</h2></div><span className="studio-tag">{catalog.length}</span></div>
    <form className="studio-library-search" onSubmit={onSearchSubmit}><label className="studio-search">Find a LoRA<input disabled={libraryRefreshing} value={search} onChange={(event) => onSearch(event.target.value)} placeholder="Search your library…" /></label><button type="submit" disabled={catalogLoading}>Search</button></form>
    <p className="studio-catalogue-hint">Family and role labels are folder hints, not verified architecture or export support. Preparing a stack checks the actual files.</p>
    {catalogueStatus === "refreshed" && <p className="studio-catalogue-hint">Current files reflects the last successful refresh. Refresh again after adding, moving or removing LoRA files.</p>}
    {catalogueStatus === "not_refreshed" && <p className="studio-help">Choose “Refresh library” to check current files. Until then, older records are available under All catalogue entries.</p>}
    {libraryPresence === "missing" && <p className="studio-help">These files were missing at the last refresh. Their catalogue and saved history remain; exports require the source file to be available again.</p>}
    <label className="studio-check"><input type="checkbox" checked={showAll} onChange={(event) => onShowAll(event.target.checked)} />Show other model families</label>
    {hiddenCount > 0 && !showAll && <p className="studio-help">{hiddenCount} other model families hidden on this page.</p>}
    {catalogLoading && <p role="status">Loading your library…</p>}{catalogError && <p role="alert">{catalogError}</p>}
    {!catalogLoading && !catalog.length && catalogueStatus !== "not_refreshed" && <p className="studio-help">No matching LoRAs. Try changing the library filters.</p>}
    <div className="studio-library-list">{catalog.map((item) => <LibraryItem key={item.stable_id} apiBase={apiBase} item={item} picked={selectedIds.includes(item.stable_id)} onToggle={onToggle} disabled={profileOperationBusy || Boolean(draftProfiles[item.stable_id])} />)}</div>
    <div className="studio-pagination"><button onClick={() => onPage(page - 1)} disabled={page === 0 || catalogLoading} aria-label="Previous library page">←</button><span>{page + 1} / {pages}</span><button onClick={() => onPage(page + 1)} disabled={page + 1 >= pages || catalogLoading} aria-label="Next library page">→</button></div>
  </aside><div className="studio-canvas"><section className="studio-panel studio-stack" aria-label="Selected stack"><div className="studio-section-heading"><div><span className="studio-eyebrow">Build your composition</span><h2>Selected stack <span>{selectedItems.length}</span></h2></div><button className="studio-text-button" onClick={onClear} disabled={!selectedItems.length || operationBusy || dirty}>Clear stack</button></div>
    {selectedItems.length ? <div className="studio-stack-items">{selectedItems.map((item, index) => <div key={item.stable_id} className={`studio-stack-item ${selected?.stable_id === item.stable_id ? "is-active" : ""}`}><button onClick={() => { setFocusedId(item.stable_id); setSelectedSlot(0); }}><i style={{ background: COLOURS[index % COLOURS.length] }} /><span><small>Loader {index + 1} · {item.role || "Role unknown"}</small><strong>{nameOf(item)}</strong></span></button><button disabled={operationBusy || Boolean(draftProfiles[item.stable_id])} onClick={() => onRemove(item.stable_id)} aria-label={`Remove ${nameOf(item)}`}>×</button></div>)}</div> : <p className="studio-help">Choose LoRAs from the library. Each gets its own complete block vector in the order you add it.</p>}
    <div className="studio-action-row"><p>Target: <strong>FLUX.1 dev · Inspire block loader</strong><span>Conditional mapping · checkpoint not verified</span></p><button className="studio-primary" onClick={onCalculate} disabled={!selectedItems.length || operationBusy || dirty}>{loading ? "Preparing…" : "Prepare block values"}</button></div>
    {dirty && <p className="studio-help">Personal draft changes are visible below. Save or discard them before preparing a new export.</p>}
    {error && <p role="alert" className="studio-alert">{error}</p>}
    {result && <details className="studio-evidence"><summary>Calculation details</summary><p>{result.compatible === false ? "Needs attention" : "Architecture compatibility checked"} · {result.validated_base_model} · {result.validated_layout}</p>{[...(result.reasons || []), ...(result.warnings || [])].map((text, index) => <p key={index}>{typeof text === "string" ? text : text?.reason_detail || text?.reason || "A selected LoRA could not be prepared."}</p>)}</details>}
  </section>
  <section className="studio-panel studio-editor"><div className="studio-section-heading"><div><span className="studio-eyebrow">Individual block weights</span><h2>{selected ? nameOf(selected) : "Make room for every LoRA"}</h2></div><span className="studio-tag">{variants.records[selected?.stable_id]?.draft ? "Personal draft" : variants.records[selected?.stable_id]?.selected?.name || "Server result"}</span></div>
    {contract.ready ? <><p className="studio-help">Select a bar to inspect its exact signed value. Bar height shows magnitude; coral marks negative weights. Export follows the verified loader order, which can differ for sparse adapters.</p><BlockChart contract={contract} selectedSlot={selectedSlot} onSelect={setSelectedSlot} /><div className="studio-inspector"><div><span className="studio-eyebrow">Selected slot</span><strong>{slot.label}</strong></div><label>Exact value<input type="number" readOnly value={slot.value} /></label><p>Default stays unchanged. Use the variants panel to make and save a personal revision.</p></div></> : <div className="studio-editor-empty"><span className="studio-orbit" aria-hidden="true">▥</span><h3>{selected ? "Your block workspace" : "Start with your first LoRA"}</h3><p>{selected ? result ? contract.reason : "Prepare the stack to check its loader mapping and inspect individual blocks." : "Select a person, clothing, style or another LoRA from your library."}</p></div>}
    {selected && <VersionPanel id={selected.stable_id} record={variants.records[selected.stable_id]} slotIndex={selectedSlot} actions={variants} loading={operationBusy} />}
  </section><Comparison items={selectedItems} getDisplayContract={getDisplayContract} />
  <MeasurementPanel apiBase={apiBase} selectedItems={selectedItems} versionIds={versionIds} result={result} dirty={dirty} loading={loading || baseProfileBusy} experimentBusy={experimentBusy} onJobChange={setAnalysisJob} onInvalidatePrepared={onInvalidatePrepared} />
  <ExperimentPanel apiBase={apiBase} job={analysisJob} selectedItems={selectedItems} versionIds={versionIds} result={result} dirty={dirty} loading={loading || baseProfileBusy} currentRecipe={currentRecipe} onBusyChange={setExperimentBusy} onRestore={onRestoreComposition} onInvalidatePrepared={onInvalidatePrepared} selectedId={selected?.stable_id} selectedSlot={selectedSlot} />
  {result && <section className="studio-exports" aria-label="Full loader vectors"><div className="studio-section-heading"><div><span className="studio-eyebrow">Take it into ComfyUI</span><h2>One full vector per loader</h2></div></div>{selectedItems.map((item, index) => <LoaderCard key={item.stable_id} item={item} index={index} payload={computedById.get(item.stable_id)} loading={operationBusy} />)}</section>}
  <CompositionPanel apiBase={apiBase} selectedIds={selectedIds} versionIds={versionIds} result={result} dirty={dirty} loading={operationBusy} onBusyChange={setRecipeBusy} currentRecipe={currentRecipe} onSaved={onRecipeSaved} onRestore={onRestoreComposition} onVersionChange={onVersionChange} onInvalidatePrepared={onInvalidatePrepared} />
  </div></div>;
}
