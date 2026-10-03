import { useEffect, useRef, useState } from "react";
import "./ExperimentPanel.css";

async function post(url, body) {
  const uncertain = url.endsWith("/save") ? "The save outcome may be unknown; retry with the same details or recover the recipe from history." : "No preview was received. Try previewing again.";
  let response;
  try { response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
  catch { throw new Error(`The connection was interrupted. ${uncertain}`); }
  let data;
  try { data = await response.json(); } catch { throw new Error(`The response could not be read. ${uncertain}`); }
  if (!response.ok) { const error = new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || `Experiment request failed (${response.status}).`); error.status = response.status; throw error; }
  return data;
}

function validPreview(data, entries, jobId, digest) {
  const policy = data?.policy_preview;
  const finiteVector = (values) => Array.isArray(values) && values.length === 58 && values.every(Number.isFinite);
  return data?.job_id === jobId && data.input_preparation_digest === digest && typeof data.proposal_digest === "string" && typeof data.can_save === "boolean" && policy?.status === "experimental_preview" && policy.calibrated === false && Array.isArray(policy.entries) && policy.entries.length === entries.length && policy.entries.every((entry, index) => entry.stable_id === entries[index].stable_id && entry.profile_version_id === entries[index].profile_version_id && finiteVector(entry.before_values) && finiteVector(entry.values)) && Array.isArray(policy.changes) && policy.changes.every((change) => entries.some((entry) => entry.stable_id === change.stable_id) && Number.isInteger(change.slot_index) && change.slot_index >= 0 && change.slot_index < 58 && typeof change.slot_label === "string" && [change.before, change.value, change.min, change.max].every(Number.isFinite) && change.min <= change.value && change.value <= change.max);
}

export default function ExperimentPanel({ apiBase, job, selectedItems, versionIds, result, dirty, loading, currentRecipe, onBusyChange, onRestore, onInvalidatePrepared, selectedId, selectedSlot }) {
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const [choices, setChoices] = useState({});
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const retry = useRef(null);
  const priorities = Object.fromEntries(selectedItems.map((item) => [item.stable_id, choices[item.stable_id] ?? 1]));
  const entries = selectedItems.map((item) => ({ stable_id: item.stable_id, profile_version_id: versionIds[item.stable_id] }));
  const context = JSON.stringify([job?.job_id, result?.preparation_digest, entries, priorities, dirty]);
  const [seenContext, setSeenContext] = useState(context);
  if (seenContext !== context) { setSeenContext(context); setPreview(null); }
  const eligible = job?.status === "complete" && job.preparation_digest === result?.preparation_digest && JSON.stringify(job.entries) === JSON.stringify(entries) && selectedItems.length >= 2 && selectedItems.length <= 8 && !dirty && !loading;
  const shown = eligible && preview?.context === context ? preview : null;
  const policy = shown?.policy_preview;
  const focused = policy?.entries.find((entry) => entry.stable_id === selectedId);
  const slotChange = policy?.changes.find((change) => change.stable_id === selectedId && change.slot_index === selectedSlot);
  const selectedName = selectedItems.find((item) => item.stable_id === selectedId)?.filename || selectedId;
  const nameOf = (id) => selectedItems.find((item) => item.stable_id === id)?.filename || id;
  function changePriority(id, value) { setChoices((previous) => ({ ...previous, [id]: value })); setPreview(null); setError(""); setMessage(""); retry.current = null; }
  async function propose() {
    if (!eligible || busy) return;
    setBusy(true); onBusyChange(true); setPreview(null); setError(""); setMessage(""); retry.current = null;
    try {
      const data = await post(`${apiBase}/experiments/preview`, { job_id: job.job_id, priorities, expected_preparation_digest: result.preparation_digest });
      if (!validPreview(data, entries, job.job_id, result.preparation_digest)) throw new Error("The experiment response does not match the saved composition.");
      if (alive.current) setPreview({ ...data, context });
    } catch (failure) { if (alive.current) { setError(failure.message); if (failure.status === 409 || failure.status === 422) onInvalidatePrepared(); } }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }
  async function save() {
    if (!shown?.can_save || !policy.changes.length || !name.trim() || busy) return;
    const payload = { job_id: job.job_id, priorities, expected_preparation_digest: result.preparation_digest, expected_proposal_digest: shown.proposal_digest, name: name.trim(), ...(currentRecipe ? { parent_version_id: currentRecipe.version_id } : {}) };
    const signature = JSON.stringify(payload);
    if (retry.current?.signature !== signature) retry.current = { signature, key: crypto.randomUUID() };
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    try {
      const saved = await post(`${apiBase}/experiments/save`, { ...payload, idempotency_key: retry.current.key });
      if (!alive.current) return;
      if (saved.status === "no_changes") { setMessage(saved.reason || "No changes were saved."); setPreview(null); return; }
      if (saved.status !== "saved" || !saved.composition?.version_id || !Array.isArray(saved.composition.entries)) throw new Error("The save response was incomplete. Retry with the same details or recover the recipe from history.");
      onRestore(saved.composition);
    } catch (failure) {
      if (alive.current) {
        setError(failure.message);
        if (failure.status === 409 || failure.status === 422) { setPreview(null); onInvalidatePrepared(); }
      }
    } finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }
  return <section className="studio-panel studio-experiments" aria-label="Guided block experiment"><div className="studio-section-heading"><div><span className="studio-eyebrow">Try a measured adjustment</span><h2>Guided block experiment</h2></div><span className="studio-tag">Experimental</span></div>
    <p className="studio-help">Choose which LoRAs should give way when measured contributions overlap. All start at Normal; priorities are your choice. This is an uncalibrated parameter experiment, not an image-quality or compatibility guarantee.</p>
    <div className="studio-priorities">{selectedItems.map((item) => <label key={item.stable_id}>{nameOf(item.stable_id)}<select aria-label={`Priority for ${nameOf(item.stable_id)}`} disabled={busy || loading} value={priorities[item.stable_id]} onChange={(event) => changePriority(item.stable_id, Number(event.target.value))}><option value={0}>Flexible · may be reduced</option><option value={1}>Normal</option><option value={2}>Protect · higher priority</option></select></label>)}</div>
    <p className="studio-help">Protect retains its current values. Normal can yield to Protect; Flexible can yield to either. Equal priorities and BASE remain unchanged.</p>
    <button className="studio-primary" disabled={!eligible || busy} onClick={propose}>{busy ? "Working…" : "Preview block experiment"}</button>
    {!eligible && <p className="studio-help">Prepare and measure two to eight saved profiles above before previewing an experiment.</p>}
    {shown && <div className="studio-experiment-preview"><h3>{policy.changes.length ? `${policy.changes.length} block changes proposed` : "No block changes proposed"}</h3><p className="studio-help">{policy.changes.length ? "Review the proposed numbers before saving a separate version. Existing Defaults and revisions stay intact." : "The current priorities and measured criteria produced no changes. This does not prove that the LoRAs are compatible or that the image will improve."}</p>
      {focused && <div className="studio-experiment-focus"><span>{selectedName} · {slotChange?.slot_label || (selectedSlot === 0 ? "BASE" : selectedSlot <= 19 ? `DOUBLE ${selectedSlot - 1}` : `SINGLE ${selectedSlot - 20}`)}</span><strong>{focused.before_values[selectedSlot]} → {focused.values[selectedSlot]}</strong><small>{slotChange ? `Trial interval ${slotChange.min} to ${slotChange.max}. ${slotChange.basis}` : "No change to this selected slot."}</small></div>}
      <p className="studio-help">{shown.ab_handling}</p>
      {policy.changes.length > 0 && <details className="studio-experiment-changes"><summary>Review every proposed block change</summary><div className="studio-measurement-table" tabIndex={0} role="region" aria-label="Proposed block changes"><table><thead><tr><th>LoRA / block</th><th>Before → proposed</th><th>Trial min / max</th><th>Reason</th></tr></thead><tbody>{policy.changes.map((change) => <tr key={`${change.stable_id}-${change.slot_index}`}><th>{nameOf(change.stable_id)}<small>{change.slot_label}</small></th><td>{change.before} → {change.value}</td><td>{change.min} / {change.max}</td><td>{change.reason}<small>{change.basis}</small></td></tr>)}</tbody></table></div></details>}
      <details className="studio-experiment-changes"><summary>Policy limits and unchanged blocks</summary>{(policy.limitations || []).map((text, index) => <p className="studio-help" key={index}>{text}</p>)}<p className="studio-help">Maximum reduction: {Number(policy.constants?.max_reduction) * 100}%. Positive-pressure threshold: {policy.constants?.pressure_threshold}. These constants have not been calibrated against image tests. {policy.blocks?.filter((block) => block.reverted).length || 0} block adjustments were reverted because the proposed reduction increased signed combined energy.</p></details>
      <div className="studio-experiment-save"><label>Experiment name<input maxLength={200} disabled={busy || loading} value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Protect the portrait" /></label><button className="studio-primary" disabled={busy || loading || !shown.can_save || !policy.changes.length || !name.trim()} onClick={save}>Save as new experiment</button></div>
      <p className="studio-help">Saving creates new personal revisions and a recipe together. It loads that saved stack; prepare it again before copying any loader vectors.</p>
    </div>}
    {message && <p role="status">{message}</p>}{error && <p role="alert" className="studio-alert">{error}</p>}
  </section>;
}
