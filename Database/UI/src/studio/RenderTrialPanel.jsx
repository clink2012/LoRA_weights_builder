import { useEffect, useRef, useState } from "react";
import "./RenderTrialPanel.css";

async function request(url, options) {
  let response;
  try { response = await fetch(url, options); } catch { throw new Error("Connection interrupted. Retry with the same details; previous saves are preserved."); }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not read the render history.");
  return data;
}
const jsonPost = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const emptyGeneration = { checkpoint: "", checkpoint_sha256: null, positive_prompt: "", negative_prompt: "", seed: "", width: 896, height: 1152, steps: 20, sampler: "dpmpp_2m", scheduler: "sgm_uniform", guidance: 3.5, denoise: 1, stage: "first_pass" };
const emptyAssessment = { identity: "not_assessed", effect: "not_assessed", colour: "not_assessed", outcome: "unresolved", notes: "", regressions: "" };
const choices = {
  identity: [["not_assessed", "Not assessed"], ["retained", "Retained"], ["changed", "Changed"], ["uncertain", "Uncertain"], ["not_applicable", "Not applicable"]],
  effect: [["not_assessed", "Not assessed"], ["recovered", "Recovered"], ["partial", "Partly recovered"], ["absent", "Absent"]],
  colour: [["not_assessed", "Not assessed"], ["correct", "Correct"], ["partial", "Partly correct"], ["absent", "Absent"], ["not_applicable", "Not applicable"]],
  outcome: [["unresolved", "Unresolved"], ["accepted", "Accepted repair"], ["partial", "Partial improvement"], ["no_improvement", "No improvement"], ["worse", "Worse"]],
};
const labelOf = (field, value) => choices[field].find(([key]) => key === value)?.[1] || value;

function TrialDetail({ apiBase, trial, onChange }) {
  const endpoint = `${apiBase}/render-trials/${encodeURIComponent(trial.trial_id)}`;
  const latest = trial.assessments.at(-1);
  const [assessment, setAssessment] = useState(latest?.assessment || emptyAssessment);
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const retry = useRef(null);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  async function mutate(operation, success) {
    setBusy(true); setError(""); setMessage("");
    try { const data = await operation(); if (alive.current) { onChange(data); setMessage(success); } }
    catch (failure) { if (alive.current) setError(failure.message); }
    finally { if (alive.current) setBusy(false); }
  }
  function saveAssessment() {
    const payload = { assessment, expected_assessment_id: latest?.assessment_id || null };
    const signature = JSON.stringify(payload);
    if (retry.current?.signature !== signature) retry.current = { signature, key: crypto.randomUUID() };
    mutate(() => request(`${endpoint}/assessments`, jsonPost({ ...payload, idempotency_key: retry.current.key })), "Assessment saved. Earlier answers remain in history.");
  }
  return <div className="studio-trial-detail">
    <h3>{trial.name}</h3><p className="studio-help">Recipe: {trial.receipt.composition.name}. Intended contributions: {trial.receipt.criteria}</p>
    <p className="studio-help">Generation settings were entered by you. Uploaded metadata is retained as evidence; matching it to the recipe is not automatic.</p>
    <details><summary>Recorded settings and exact loader values</summary><pre>{JSON.stringify({ generation: trial.receipt.declared_generation, recipe: trial.receipt.composition.historical_snapshot }, null, 2)}</pre></details>
    <div className="studio-trial-images">{trial.evidence.map((item) => <figure key={item.evidence_id}><a href={`${endpoint}/evidence/${encodeURIComponent(item.evidence_id)}`} target="_blank" rel="noreferrer"><img src={`${endpoint}/evidence/${encodeURIComponent(item.evidence_id)}`} alt={`Render evidence: ${item.filename}`} /></a><figcaption>{item.filename}<small>SHA256 {item.sha256}</small><small>Recipe match unverified</small></figcaption></figure>)}</div>
    <div className="studio-trial-fields"><label>Rendered PNG<input type="file" accept="image/png" disabled={busy} onChange={(event) => setFile(event.target.files?.[0] || null)} /></label><button disabled={busy || !file} onClick={() => mutate(() => request(`${endpoint}/evidence?filename=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": "image/png" }, body: file }), "PNG preserved with its content hash.")}>Attach PNG</button></div>
    <p className="studio-help">PNG limit: 16 MiB. Files are stored with the local database. ComfyUI is not opened or queued.</p>
    <fieldset disabled={busy}><legend>Your visual assessment</legend><div className="studio-trial-fields">{[["identity", "Identity / key features"], ["effect", "Intended effect"], ["colour", "Colour / material"], ["outcome", "Overall result"]].map(([field, label]) => <label key={field}>{label}<select value={assessment[field]} onChange={(event) => setAssessment({ ...assessment, [field]: event.target.value })}>{choices[field].map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select></label>)}</div>
      <label>Regressions<textarea maxLength={4000} value={assessment.regressions} onChange={(event) => setAssessment({ ...assessment, regressions: event.target.value })} placeholder="Other intended contributions lost or new defects" /></label>
      <label>Assessment notes<textarea maxLength={4000} value={assessment.notes} onChange={(event) => setAssessment({ ...assessment, notes: event.target.value })} /></label>
      <button disabled={!trial.evidence.length} onClick={saveAssessment}>{latest ? "Save corrected assessment" : "Save assessment"}</button>
    </fieldset>
    {!trial.evidence.length && <p className="studio-help">Attach the rendered PNG before saving an assessment.</p>}
    <details><summary>Assessment history ({trial.assessments.length})</summary>{trial.assessments.map((item, index) => <article key={item.assessment_id}><strong>{index + 1}. {labelOf("outcome", item.assessment.outcome)}</strong><p>Identity: {labelOf("identity", item.assessment.identity)} · Effect: {labelOf("effect", item.assessment.effect)} · Colour: {labelOf("colour", item.assessment.colour)}</p><p>{item.assessment.notes}</p>{item.assessment.regressions && <p>Regressions: {item.assessment.regressions}</p>}</article>)}</details>
    {message && <p role="status">{message}</p>}{error && <p role="alert" className="studio-alert">{error}</p>}
  </div>;
}

export default function RenderTrialPanel({ apiBase, currentRecipe, selectedIds, versionIds, dirty, loading }) {
  const endpoint = `${apiBase}/render-trials`;
  const [trials, setTrials] = useState([]);
  const [trial, setTrial] = useState(null);
  const [generation, setGeneration] = useState(emptyGeneration);
  const [name, setName] = useState("");
  const [criteria, setCriteria] = useState("");
  const [baseline, setBaseline] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const retry = useRef(null);
  const alive = useRef(true);
  const loadSerial = useRef(0);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    let active = true;
    request(`${endpoint}?limit=50&offset=${offset}`).then((data) => { if (active) setTrials(data.trials); }).catch((failure) => { if (active) setError(failure.message); });
    return () => { active = false; };
  }, [endpoint, offset]);
  const entries = selectedIds.map((id) => ({ stable_id: id, profile_version_id: versionIds[id] }));
  const recipeMatches = currentRecipe && Array.isArray(currentRecipe.entries) && entries.length === currentRecipe.entries.length && entries.every((entry, index) => entry.stable_id === currentRecipe.entries[index].stable_id && entry.profile_version_id === currentRecipe.entries[index].profile_version_id);
  async function loadTrial(id, useAsBaseline = false) {
    const serial = ++loadSerial.current;
    setError(""); setBusy(true); setTrial(null);
    try {
      const data = await request(`${endpoint}/${encodeURIComponent(id)}`);
      if (alive.current && serial === loadSerial.current) { setTrial(data); if (useAsBaseline) { setGeneration(data.receipt.declared_generation); setCriteria(data.receipt.criteria); setBaseline(id); } }
    } catch (failure) { if (alive.current && serial === loadSerial.current) setError(failure.message); }
    finally { if (alive.current && serial === loadSerial.current) setBusy(false); }
  }
  async function create() {
    const payload = { name: name.trim(), composition_version_id: currentRecipe.version_id, generation, criteria, baseline_trial_id: baseline || null };
    const signature = JSON.stringify(payload);
    if (retry.current?.signature !== signature) retry.current = { signature, key: crypto.randomUUID() };
    setBusy(true); setError("");
    try {
      const data = await request(endpoint, jsonPost({ ...payload, idempotency_key: retry.current.key }));
      if (alive.current) { setTrial(data); setTrials((old) => [data, ...old.filter((item) => item.trial_id !== data.trial_id)].slice(0, 50)); }
    } catch (failure) { if (alive.current) setError(failure.message); }
    finally { if (alive.current) setBusy(false); }
  }
  function change(field, value) { setGeneration({ ...generation, [field]: value }); }
  return <section className="studio-panel studio-render-trials" aria-label="Render trials"><details><summary>Render trials and visual results</summary>
    <p className="studio-help">Keep the exact saved recipe, render settings, image and your judgement together. Failed and unresolved trials are useful evidence too.</p>
    <div className="studio-trial-fields"><label>Saved trial<select aria-label="Saved trial" disabled={busy} value={trial?.trial_id || ""} onChange={(event) => { if (event.target.value) loadTrial(event.target.value); else setTrial(null); }}><option value="">Choose a trial…</option>{trial && !trials.some((item) => item.trial_id === trial.trial_id) && <option value={trial.trial_id}>{trial.name} · currently open</option>}{trials.map((item) => <option key={item.trial_id} value={item.trial_id}>{item.name} · {new Date(item.created_at).toLocaleString()}</option>)}</select></label><button disabled={busy || !trial} onClick={() => loadTrial(trial.trial_id, true)}>Use settings as comparison baseline</button><button disabled={busy || !trial} onClick={() => loadTrial(trial.trial_id)}>Reload trial</button></div>
    <div className="studio-trial-pages"><button disabled={busy || offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Newer trials</button><button disabled={busy || trials.length < 50} onClick={() => setOffset(offset + 50)}>Older trials</button></div>
    <details><summary>Record a new trial</summary><p className="studio-help">Saved recipe: {currentRecipe?.name || "none selected"}. Save the current stack as a recipe first.</p>
      <fieldset disabled={busy || loading}><div className="studio-trial-fields"><label>Trial name<input maxLength={200} value={name} onChange={(event) => setName(event.target.value)} /></label><label>Checkpoint<input maxLength={500} value={generation.checkpoint} onChange={(event) => change("checkpoint", event.target.value)} /></label><label>Checkpoint SHA256 (optional)<input maxLength={64} value={generation.checkpoint_sha256 || ""} onChange={(event) => change("checkpoint_sha256", event.target.value || null)} /></label></div>
      <label>Intended contributions<textarea maxLength={2000} value={criteria} onChange={(event) => setCriteria(event.target.value)} placeholder="e.g. Retain Sabrina's features and recover dark polka-dot tights" /></label>
      <label>Positive prompt<textarea maxLength={20000} value={generation.positive_prompt} onChange={(event) => change("positive_prompt", event.target.value)} /></label><label>Negative prompt<textarea maxLength={20000} value={generation.negative_prompt} onChange={(event) => change("negative_prompt", event.target.value)} /></label>
      <div className="studio-trial-fields"><label>Fixed seed<input inputMode="numeric" value={generation.seed} onChange={(event) => change("seed", event.target.value)} /></label>{["width", "height", "steps", "guidance", "denoise"].map((field) => <label key={field}>{field[0].toUpperCase() + field.slice(1)}<input type="number" step={field === "guidance" || field === "denoise" ? "any" : "1"} value={generation[field]} onChange={(event) => change(field, Number(event.target.value))} /></label>)}{["sampler", "scheduler"].map((field) => <label key={field}>{field[0].toUpperCase() + field.slice(1)}<input value={generation[field]} onChange={(event) => change(field, event.target.value)} /></label>)}<label>Render stage<select value={generation.stage} onChange={(event) => change("stage", event.target.value)}><option value="first_pass">First pass</option><option value="final">Final result</option></select></label></div>
      {baseline && <p>Comparison baseline selected. Generation settings must match it.<button onClick={() => setBaseline("")}>Clear baseline</button></p>}
      <button className="studio-primary" disabled={!recipeMatches || dirty || !name.trim() || !criteria.trim() || !generation.checkpoint || !generation.positive_prompt.trim() || !generation.seed} onClick={create}>Save trial record</button></fieldset>
      {(!recipeMatches || dirty) && <p className="studio-help">Save the current selection and profile values as a recipe before recording this trial.</p>}
    </details>
    {trial && <TrialDetail key={trial.trial_id} apiBase={apiBase} trial={trial} onChange={setTrial} />}
    {error && <p role="alert" className="studio-alert">{error}</p>}
  </details></section>;
}
