import { useLayoutEffect, useRef, useState } from "react";

function ExperimentEditor({ record, slotIndex, onEdit, disabled }) {
  const [symbol, setSymbol] = useState("A");
  const [minimum, setMinimum] = useState("0");
  const [maximum, setMaximum] = useState("1");
  const [error, setError] = useState("");
  const snapshot = record.draft || record.computed || record.selected;
  const slot = record.selected.binding.slots[slotIndex];
  function apply() {
    const min = Number(minimum), max = Number(maximum), value = snapshot.values[slotIndex];
    if (!minimum.trim() || !maximum.trim() || !Number.isFinite(min) || !Number.isFinite(max) || min > max || value < min || value > max) { setError("Enter finite minimum and maximum values that include this block's exact value."); return; }
    const ab = Object.fromEntries(Object.entries(snapshot.ab || {}).filter(([key, experiment]) => key !== symbol && !experiment.slot_labels.includes(slot.label)));
    ab[symbol] = { slot_labels: [slot.label], min, max, value, basis: "Personal trial bounds; not measured image-quality limits or loader limits." };
    onEdit({ ab }); setError("");
  }
  return <details className="studio-experiment"><summary>Personal A/B experiment</summary><p className="studio-help">Choose a trial range for {slot.label}. These are your experiment bounds, not a proven recommendation. Numeric export uses the current exact value; no A/B template is exported yet.</p>
    <div className="studio-experiment-fields"><label>Symbol<select aria-label="Experiment symbol" disabled={disabled} value={symbol} onChange={(event) => setSymbol(event.target.value)}><option>A</option><option>B</option></select></label><label>Minimum<input type="number" step="any" disabled={disabled} value={minimum} onChange={(event) => setMinimum(event.target.value)} /></label><label>Maximum<input type="number" step="any" disabled={disabled} value={maximum} onChange={(event) => setMaximum(event.target.value)} /></label><button onClick={apply} disabled={disabled || record.busy}>Set trial range</button></div>
    {error && <p role="alert">{error}</p>}
    {Object.entries(snapshot.ab || {}).map(([key, experiment]) => <div key={key} className="studio-experiment-summary"><span>{key} · {experiment.slot_labels.join(", ")} · {experiment.min} to {experiment.max} · current {experiment.value}</span><button disabled={disabled || record.busy} onClick={() => onEdit({ ab: Object.fromEntries(Object.entries(snapshot.ab).filter(([symbolName]) => symbolName !== key)) })}>Remove {key}</button></div>)}
  </details>;
}

function ExactEditor({ value, onChange, disabled, inputRef, onBeforeChange, label = "New exact value", actionLabel = "Apply block value" }) {
  const [text, setText] = useState(String(value));
  const [error, setError] = useState("");
  function commit() {
    const number = Number(text);
    if (!text.trim() || !Number.isFinite(number)) { setError("Enter a finite number."); return; }
    setError(""); onBeforeChange?.(); onChange(number);
  }
  return <div className="studio-exact-edit"><label>{label}<input ref={inputRef} type="text" inputMode="decimal" value={text} onChange={(event) => { setText(event.target.value); setError(""); }} disabled={disabled} /></label><button onClick={commit} disabled={disabled || text === String(value)}>{actionLabel}</button>{error && <p role="alert">{error}</p>}</div>;
}

function LoraSettings({ snapshot, disabled, onEdit }) {
  const strengthInput = useRef(null);
  const focusRequested = useRef(false);
  useLayoutEffect(() => {
    if (focusRequested.current && !disabled && strengthInput.current) {
      strengthInput.current.focus();
      focusRequested.current = false;
    }
  }, [snapshot, disabled]);
  const roles = ["person", "character", "clothing", "pose", "style", "environment", "utility", "other"];
  if (!roles.includes(snapshot.settings.role)) roles.unshift(snapshot.settings.role);
  return <details className="studio-experiment"><summary>LoRA role and supporting settings</summary><p className="studio-help">A role describes your intent. Changing it does not automatically rebalance blocks in this version.</p><label>LoRA role<select aria-label="LoRA role" disabled={disabled} value={snapshot.settings.role} onChange={(event) => onEdit({ settings: { ...snapshot.settings, role: event.target.value } })}>{roles.map((role) => <option key={role} value={role}>{role}</option>)}</select></label><ExactEditor key={snapshot.settings.strength_model} value={snapshot.settings.strength_model} inputRef={strengthInput} onBeforeChange={() => { focusRequested.current = true; }} label="Model strength" actionLabel="Apply model strength" disabled={disabled} onChange={(value) => onEdit({ settings: { ...snapshot.settings, strength_model: value } })} /><p className="studio-help">CLIP: model-only adapter; separate CLIP editing is unavailable.</p></details>;
}

export default function VersionPanel({ id, record, slotIndex, actions, loading }) {
  const editorInput = useRef(null);
  const openTrigger = useRef(null);
  const focusRequestedFor = useRef(null);
  useLayoutEffect(() => {
    if (focusRequestedFor.current !== id || loading || record?.busy) return;
    const target = record?.selected ? editorInput.current : record?.error ? openTrigger.current : null;
    if (target) { target.focus(); focusRequestedFor.current = null; }
  }, [id, record, loading]);
  if (!record?.selected) return <section className="studio-variant-panel"><h3>Keep your Default. Make it yours.</h3><p className="studio-help">Capture Default from the current file to create personal variants. Each save becomes a new entry; earlier versions remain available.</p><button ref={openTrigger} className="studio-primary" disabled={loading || record?.busy} onClick={() => { focusRequestedFor.current = id; actions.open(id); }}>{record?.busy ? "Opening history…" : "Open variants & history"}</button>{record?.error && <p role="alert">{record.error}</p>}</section>;
  const snapshot = record.draft || record.computed || record.selected;
  const slot = record.selected.binding.slots[slotIndex] || record.selected.binding.slots[0];
  const index = record.selected.binding.slots.indexOf(slot);
  return <section className="studio-variant-panel" aria-label="Variants and history"><div className="studio-section-heading"><div><span className="studio-eyebrow">Default is preserved</span><h3>{record.draft ? "Personal draft" : record.computed?.name || record.selected.name}</h3></div><span className="studio-tag">{record.draft ? "Not saved" : record.computed ? "Computed baseline · not a personal recipe" : `Version ${record.selected.sequence}`}</span></div>
    <button disabled={loading || record.busy || Boolean(record.draft)} onClick={() => actions.open(id)}>Refresh variants & history</button>
    <div className="studio-version-layout"><div className="studio-version-editor"><p className="studio-help">Editing {slot.label}. Save a named revision, then prepare its loader values again before copying.</p>
      <ExactEditor key={`${record.selected.version_id}:${index}:${snapshot.values[index]}`} value={snapshot.values[index]} inputRef={editorInput} onBeforeChange={() => { focusRequestedFor.current = id; }} onChange={(value) => actions.editValue(id, index, value)} disabled={loading || record.busy} />
      <ExperimentEditor key={`${record.selected.version_id}:${index}`} record={record} slotIndex={index} disabled={loading || record.busy} onEdit={(update) => actions.edit(id, update)} />
      <LoraSettings snapshot={snapshot} disabled={loading || record.busy} onEdit={(update) => actions.edit(id, update)} />
      {record.draft && <div className="studio-save-revision"><label>Revision name<input disabled={loading || record.busy} value={record.name} maxLength={200} placeholder="e.g. Softer clothing" onChange={(event) => actions.setName(id, event.target.value)} /></label><div><button className="studio-primary" onClick={() => actions.save(id)} disabled={loading || record.busy || !record.name?.trim() || record.name.trim().toLowerCase() === "default"}>{record.busy ? "Saving…" : "Save new revision"}</button><button onClick={() => actions.discard(id)} disabled={loading || record.busy}>Discard draft</button></div></div>}
    </div><div className="studio-history"><h4>History</h4><p className="studio-help">Choosing an earlier version preserves everything saved after it.</p><div className="studio-history-list">{[...record.versions].reverse().map((version) => <button key={version.version_id} disabled={loading || record.busy || Boolean(record.draft)} aria-pressed={record.selected.version_id === version.version_id} onClick={() => actions.select(id, version.version_id)}><strong>{version.name}</strong><small>Version {version.sequence} · {version.kind === "default" ? "Original Default" : "Personal revision"}</small></button>)}</div></div></div>
    {record.error && <p className="studio-alert" role="alert">{record.error}</p>}
  </section>;
}
