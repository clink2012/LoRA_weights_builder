import { useEffect, useRef, useState } from "react";
import "./MeasurementPanel.css";

const TARGET = "flux1-dev-native-v1";
const ACTIVE = new Set(["queued", "running"]);
const STATUSES = new Set([...ACTIVE, "complete", "failed", "cancelled", "stale"]);
const format = (value) => value === null ? "Not defined" : value === 0 ? "0" : Number(value).toPrecision(4);
const identity = (entries, digest) => JSON.stringify([entries, digest]);

async function request(url, body, signal) {
  const response = await fetch(url, { ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }), signal });
  const data = await response.json();
  if (!response.ok) { const error = new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || `Measurement request failed (${response.status}).`); error.status = response.status; error.code = data.detail?.reason_code; throw error; }
  return data;
}

function validateJob(data, expected) {
  if (!data || typeof data.job_id !== "string" || !STATUSES.has(data.status) || data.target_contract_id !== TARGET || identity(data.entries, data.preparation_digest) !== expected) throw new Error("The measurement response does not match the requested composition.");
  return data;
}

function validMetrics(metrics, count) {
  const finiteArray = (values, nonnegative = false) => Array.isArray(values) && values.length === 58 && values.every((value) => Number.isFinite(value) && (!nonnegative || value >= 0));
  return metrics?.status === "complete" && metrics.measurement_basis === "effective_native_parameter_update" && metrics.outer_model_strength_applied === false && metrics.block_weights_applied === false && Array.isArray(metrics.slot_labels) && metrics.slot_labels.length === 58 && metrics.slot_labels.every((label) => typeof label === "string") && Array.isArray(metrics.sources) && metrics.sources.length === count && metrics.sources.every((source, index) => source.source_index === index && finiteArray(source.block_norms, true) && Number.isFinite(source.total_squared_norm) && source.total_squared_norm >= 0) && Array.isArray(metrics.pairs) && metrics.pairs.every((pair) => Number.isInteger(pair.left_index) && Number.isInteger(pair.right_index) && pair.left_index >= 0 && pair.right_index > pair.left_index && pair.right_index < count && Array.isArray(pair.block_signed_cosines) && pair.block_signed_cosines.length === 58 && pair.block_signed_cosines.every((value) => value === null || Number.isFinite(value) && Math.abs(value) <= 1));
}

function Measurements({ metrics, items }) {
  const [pairIndex, setPairIndex] = useState(0);
  const pair = metrics.pairs[pairIndex];
  const peak = Math.max(...metrics.sources.flatMap((source) => source.block_norms), 1e-20);
  const name = (index) => items[index]?.filename || items[index]?.stable_id || `LoRA ${index + 1}`;
  return <details className="studio-measurement-result"><summary>Inspect numerical measurements</summary>
    <div className="studio-measurement-totals">{metrics.sources.map((source, index) => <div key={index}><span>{name(index)}</span><strong>{format(Math.sqrt(source.total_squared_norm))}</strong><small>Full update norm</small></div>)}</div>
    <p className="studio-help">Larger norms mean a larger parameter update, not a stronger visible effect. Alignment ranges from −1 (opposite direction) to +1 (same direction); it is undefined when either update is zero. These measurements exclude your block weights and model strength.</p>
    {metrics.pairs.length > 0 && <label className="studio-measurement-pair">Compare alignment<select value={pairIndex} onChange={(event) => setPairIndex(Number(event.target.value))}>{metrics.pairs.map((entry, index) => <option key={index} value={index}>{name(entry.left_index)} ↔ {name(entry.right_index)}</option>)}</select></label>}
    <div className="studio-measurement-table" tabIndex={0} role="region" aria-label="Per-block parameter measurements"><table><thead><tr><th scope="col">Block</th>{metrics.sources.map((_, index) => <th scope="col" key={index}>{name(index)}<small>Update norm</small></th>)}{pair && <th scope="col">Signed alignment</th>}</tr></thead><tbody>{metrics.slot_labels.map((label, index) => <tr key={index}><th scope="row">{label}</th>{metrics.sources.map((source, sourceIndex) => <td key={sourceIndex}><span>{format(source.block_norms[index])}</span><i aria-hidden="true" style={{ width: `${source.block_norms[index] / peak * 100}%` }} /></td>)}{pair && <td>{format(pair.block_signed_cosines[index])}</td>}</tr>)}</tbody></table></div>
    <p className="studio-help">Parameter measurements only. No checkpoint inference, image test or semantic conflict score was performed. Values are rounded for display; this panel does not change your saved profiles or loader vectors.</p>
  </details>;
}

export default function MeasurementPanel({ apiBase, selectedItems, versionIds, result, dirty, loading, experimentBusy = false, onJobChange, onInvalidatePrepared }) {
  const [job, setJob] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const alive = useRef(true);
  const activeJob = useRef(null);
  const recoveryEpoch = useRef(0);
  const endpoint = `${apiBase}/analysis-jobs`;
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (activeJob.current) request(`${endpoint}/${encodeURIComponent(activeJob.current)}/cancel`, {}).catch(() => {});
    };
  }, [endpoint]);
  const entries = selectedItems.map((item) => ({ stable_id: item.stable_id, profile_version_id: versionIds[item.stable_id] }));
  const binding = identity(entries, result?.preparation_digest);
  const currentBinding = useRef(binding);
  useEffect(() => { currentBinding.current = binding; }, [binding]);
  const current = job?.binding === binding && !dirty && !loading;
  useEffect(() => { onJobChange?.(current && job?.status === "complete" && validMetrics(job.metrics, job.entries.length) ? job : null); }, [current, job, onJobChange]);
  const active = ACTIVE.has(job?.status);
  useEffect(() => { activeJob.current = active ? job.job_id : null; }, [active, job?.job_id]);
  const eligible = entries.length > 0 && entries.length <= 8 && entries.every((entry) => entry.profile_version_id) && Boolean(result?.preparation_digest) && !result?.starting_proposal && result.compatible !== false && !dirty && !loading;
  useEffect(() => {
    if (!eligible || experimentBusy || !/^[a-f0-9]{64}$/.test(result?.preparation_digest || "") || current || activeJob.current) return;
    const controller = new AbortController(), epoch = ++recoveryEpoch.current;
    const savedEntries = JSON.parse(binding)[0];
    request(`${endpoint}/resolve`, { entries: savedEntries, target_contract_id: TARGET, expected_preparation_digest: result.preparation_digest }, controller.signal).then((data) => {
      if (controller.signal.aborted || !alive.current || recoveryEpoch.current !== epoch || currentBinding.current !== binding) return;
      if (data.status === "not_found" && data.job === null) return;
      if (data.status !== "reused") throw new Error("Saved measurement recovery is unavailable.");
      const recovered = validateJob(data.job, binding);
      if (recovered.status !== "complete" || !validMetrics(recovered.metrics, savedEntries.length)) throw new Error("The saved measurement did not contain valid current measurements.");
      setJob({ ...recovered, binding, reused: true, items: selectedItems.map((item) => ({ ...item })) });
    }).catch((failure) => {
      if (!controller.signal.aborted && alive.current && recoveryEpoch.current === epoch && currentBinding.current === binding) {
        if (failure.status === 409 && ["selection_changed", "receipt_binding_changed"].includes(failure.code)) onInvalidatePrepared?.();
        setError(`Saved measurement lookup: ${failure.message} You can measure current sources manually.`);
      }
    });
    return () => { controller.abort(); };
    // One lookup follows each fresh saved binding. A manual measurement wins
    // over a late convenience lookup; it never starts CPU work automatically.
  }, [binding, eligible, endpoint, experimentBusy]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!job || !ACTIVE.has(job.status)) return;
    const controller = new AbortController();
    let timer;
    async function poll() {
      try {
        const data = validateJob(await request(`${endpoint}/${encodeURIComponent(job.job_id)}`, undefined, controller.signal), job.binding);
        if (controller.signal.aborted) return;
        setJob((previous) => previous?.job_id === data.job_id && ACTIVE.has(previous.status) ? { ...data, binding: previous.binding, items: previous.items } : previous);
        if (data.status === "stale" && currentBinding.current === job.binding) onInvalidatePrepared?.();
        setError("");
        if (ACTIVE.has(data.status)) timer = setTimeout(poll, 1000);
      } catch (failure) { if (!controller.signal.aborted) { if (currentBinding.current === job.binding && failure.status === 409 && ["selection_changed", "receipt_binding_changed"].includes(failure.code)) onInvalidatePrepared?.(); setError(`${failure.message} Check the job status again before starting another measurement.`); } }
    }
    timer = setTimeout(poll, 500);
    return () => { controller.abort(); clearTimeout(timer); };
    // The poll loop follows the captured job; current edits only invalidate its display authority.
  }, [endpoint, job?.job_id, retry]); // eslint-disable-line react-hooks/exhaustive-deps
  async function start() {
    if (!eligible || pending || active || experimentBusy) return;
    recoveryEpoch.current += 1;
    setPending(true); setError(""); setJob(null);
    try {
      const data = validateJob(await request(endpoint, { entries, target_contract_id: TARGET, expected_preparation_digest: result.preparation_digest }), binding);
      if (alive.current) setJob({ ...data, binding, items: selectedItems.map((item) => ({ ...item })) });
      else if (ACTIVE.has(data.status)) request(`${endpoint}/${encodeURIComponent(data.job_id)}/cancel`, {}).catch(() => {});
      if (alive.current && data.status === "stale" && currentBinding.current === binding) onInvalidatePrepared?.();
    } catch (failure) { if (alive.current) { if (currentBinding.current === binding && failure.status === 409 && ["selection_changed", "receipt_binding_changed"].includes(failure.code)) onInvalidatePrepared?.(); setError(failure.message); } }
    finally { if (alive.current) setPending(false); }
  }
  async function cancel() {
    setPending(true); setError("");
    try {
      const data = validateJob(await request(`${endpoint}/${encodeURIComponent(job.job_id)}/cancel`, {}), job.binding);
      if (alive.current) setJob((previous) => previous?.job_id === data.job_id && (ACTIVE.has(previous.status) || !ACTIVE.has(data.status)) ? { ...data, binding: previous.binding, items: previous.items } : previous);
      if (alive.current && data.status === "stale" && currentBinding.current === job.binding) onInvalidatePrepared?.();
    } catch (failure) { if (alive.current) setError(failure.message); }
    finally { if (alive.current) setPending(false); }
  }
  return <section className="studio-panel studio-measurements" aria-label="Parameter measurements"><div className="studio-section-heading"><div><span className="studio-eyebrow">Understand the source LoRAs</span><h2>Parameter measurements</h2></div><span className="studio-tag">Optional · local CPU</span></div>
    <p className="studio-help">Read the actual LoRA tensors to compare update size and direction in each block. This can take time for larger files. It does not predict which combination will make the best image.</p>
    <p className="studio-help">After fresh preparation, a matching saved measurement is recovered automatically. “Measure current sources” always performs a new measurement.</p>
    {result?.starting_proposal && <p className="studio-help">The Build proposal already measured these original sources. Save the proposal as a recipe to run further measured trials against its adjusted values.</p>}
    <div className="studio-measurement-actions"><button className="studio-primary" onClick={start} disabled={!eligible || pending || active || experimentBusy}>{pending && !job ? "Starting…" : "Measure current sources"}</button>{active && <button onClick={cancel} disabled={pending}>Cancel measurement</button>}{active && error && <button onClick={() => { setError(""); setRetry((value) => value + 1); }}>Check job status</button>}</div>
    {!eligible && <p className="studio-help">Choose up to eight LoRAs and finish any pending edits. Use “Capture missing Defaults” in Recipes below if needed, then “Prepare block values” to check the exact saved versions.</p>}
    {job && <p role="status">{job.status === "queued" ? "Queued for local analysis." : job.status === "running" ? "Reading tensors and measuring parameter updates…" : job.status === "complete" ? current ? job.reused ? "Saved measurements recovered and revalidated for the current composition." : "Measurements complete for the current saved composition." : "Measurements belong to an earlier composition." : job.reason || `Measurement ${job.status}.`}</p>}
    {job && !current && <p className="studio-help">The stack, saved versions or preparation changed. Earlier measurements are hidden; prepare and measure the current composition again.</p>}
    {error && <p role="alert" className="studio-alert">{error}</p>}
    {job?.status === "complete" && current && (validMetrics(job.metrics, job.entries.length) ? <Measurements metrics={job.metrics} items={job.items} /> : <p role="alert">The completed response did not contain valid parameter measurements.</p>)}
  </section>;
}
