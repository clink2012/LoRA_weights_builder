import { useRef, useState } from "react";
import "./MeasuredBlockChart.css";

const compact = (value) => value === 0 ? "0" : Number(value).toPrecision(4);
const groups = (label) => label === "BASE" ? "BASE" : label.startsWith("DOUBLE") ? "Double" : label.startsWith("SINGLE") ? "Single" : "Other";

export function MultiplierInput({ label, value, disabled, onSelect, onChange }) {
  const [text, setText] = useState(String(value));
  const [invalid, setInvalid] = useState(false);
  const [seenValue, setSeenValue] = useState(value);
  if (seenValue !== value) { setSeenValue(value); setText(String(value)); setInvalid(false); }
  function edit(next) {
    setText(next); setInvalid(false);
    const number = Number(next);
    if (next.trim() && Number.isFinite(number) && number !== value) onChange(number);
  }
  function check() {
    if (!text.trim() || !Number.isFinite(Number(text))) { setText(String(value)); setInvalid(true); }
  }
  return <input aria-label={`Multiplier for ${label}`} aria-invalid={invalid} title={invalid ? "Enter a finite signed number; the last valid value is retained." : `Exact multiplier for ${label}`} type="text" inputMode="decimal" value={text} disabled={disabled} onFocus={onSelect} onChange={(event) => edit(event.target.value)} onBlur={check} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); check(); } }} />;
}

export default function MeasuredBlockChart({ record, reference, guidance, suggestedValues, selectedSlot, onSelect, onEditValue, onEdit, disabled }) {
  const snapshot = record.draft || record.computed || record.selected;
  const slots = record.selected.binding.slots;
  const norms = reference.norms;
  // The saved version supplies the scale. Draft multipliers/strength never
  // renormalise the original line or make a growing bar look unchanged.
  const peak = Math.max(1e-20, ...norms, ...record.selected.values.map((value, index) => Math.abs(record.selected.settings.strength_model * value) * norms[index]));
  const strength = snapshot.settings.strength_model;
  const currentNorms = snapshot.values.map((value, index) => Math.abs(strength * value) * norms[index]);
  const buttons = useRef([]);
  const dragging = useRef(null);
  const scale = (value) => Math.min(100, value / peak * 100);
  const stateFor = (index) => norms[index] === 0 ? "inactive" : guidance?.[index]?.state || "review";
  const reasonFor = (index) => norms[index] === 0 ? "No measured update; changing a multiplier cannot add missing content." : guidance?.[index]?.reason || "Measured update size alone does not identify a safe or semantic adjustment.";

  function reset(index) { onEditValue(index, record.root.values[index]); }
  function move(event, index) {
    const next = event.key === "ArrowRight" ? Math.min(slots.length - 1, index + 1) : event.key === "ArrowLeft" ? Math.max(0, index - 1) : event.key === "Home" ? 0 : event.key === "End" ? slots.length - 1 : null;
    if (next !== null) { event.preventDefault(); onSelect(next); buttons.current[next]?.focus({ preventScroll: true }); buttons.current[next]?.scrollIntoView?.({ block: "nearest", inline: "nearest" }); }
    else if (event.key === "Delete" && !disabled) { event.preventDefault(); reset(index); }
    else if (["ArrowUp", "ArrowDown"].includes(event.key) && !disabled) {
      event.preventDefault();
      const value = snapshot.values[index] + (event.key === "ArrowUp" ? 1 : -1) * (event.shiftKey ? .1 : .01);
      if (Number.isFinite(value)) onEditValue(index, Number(value.toPrecision(15)));
    }
  }
  function drag(event, index) {
    if (disabled || norms[index] === 0 || strength === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.height) return;
    const ratio = Math.max(0, Math.min(1, (rect.bottom - event.clientY) / rect.height));
    const value = (dragging.current?.sign || (snapshot.values[index] < 0 ? -1 : 1)) * ratio * peak / (Math.abs(strength) * norms[index]);
    if (Number.isFinite(value)) onEditValue(index, value);
  }
  const index = Math.min(selectedSlot, slots.length - 1);
  const selectedGuide = guidance?.[index];
  return <div className="studio-measured-chart">
    <p className="studio-help">Original update <span className="studio-original-key">line</span> · current contribution bars · exact loader multipliers above. Scale stays fixed while editing. Model strength: {strength}.</p>
    <div className="studio-measured-legend">{[["inactive", "No measured update"], ["protected", "Protected in last computed trial"], ["suggested", "Explained trial"], ["review", "Review before experimenting"]].map(([state, label]) => <span key={state} data-guidance={state}><i />{label}</span>)}</div>
    <div className="studio-measured-scroll" tabIndex={0} role="region" aria-label="Measured original and editable current block contributions">
      <div className="studio-measured-canvas" style={{ "--measured-slots": slots.length }}>
        <div className="studio-measured-values">{slots.map((slot, i) => <MultiplierInput key={slot.label} label={slot.label} value={snapshot.values[i]} disabled={disabled} onSelect={() => onSelect(i)} onChange={(value) => onEditValue(i, value)} />)}</div>
        <div className="studio-measured-plot">
          <svg className="studio-original-line" viewBox={`0 0 ${slots.length * 50} 100`} preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" points={norms.map((norm, i) => `${i * 50 + 25},${100 - scale(norm)}`).join(" ")} /></svg>
          <div className="studio-measured-bars">{slots.map((slot, i) => <button ref={(element) => { buttons.current[i] = element; }} key={slot.label} type="button" className={`${selectedSlot === i ? "is-current" : ""} ${snapshot.values[i] < 0 ? "is-negative" : ""}`} data-guidance={stateFor(i)} disabled={disabled} aria-label={`${slot.label}: multiplier ${snapshot.values[i]}, current update ${currentNorms[i]}, original update ${norms[i]}`} aria-pressed={selectedSlot === i} tabIndex={selectedSlot === i ? 0 : -1} title={`${slot.label} · multiplier ${snapshot.values[i]} · original ${norms[i]} · current ${currentNorms[i]}. ${reasonFor(i)}`} onClick={() => onSelect(i)} onKeyDown={(event) => move(event, i)} onDoubleClick={() => reset(i)} onPointerDown={(event) => { if (event.button !== 0 || disabled) return; onSelect(i); dragging.current = { index: i, sign: snapshot.values[i] < 0 ? -1 : 1 }; event.currentTarget.setPointerCapture?.(event.pointerId); drag(event, i); }} onPointerMove={(event) => { if (dragging.current?.index === i) drag(event, i); }} onPointerUp={() => { dragging.current = null; }} onPointerCancel={() => { dragging.current = null; }}>
            <span className="studio-measured-fill" style={{ height: `${scale(currentNorms[i])}%` }} />{snapshot.values[i] < 0 && <span className="studio-measured-sign">−</span>}{currentNorms[i] > peak && <span className="studio-measured-overflow" title="Above the fixed display scale">↑</span>}
          </button>)}</div>
        </div>
        <div className="studio-measured-labels">{slots.map((slot) => <span key={slot.label}>{groups(slot.label)}<small>{slot.label.split(" ").at(-1)}</small></span>)}</div>
      </div>
    </div>
    <div className="studio-measured-inspector"><strong>{slots[index].label}</strong><span>Original update {compact(norms[index])} · current {compact(currentNorms[index])}{norms[index] ? ` · ${(currentNorms[index] / norms[index] * 100).toFixed(1)}% of original` : " · zero original update"}</span><span>{reasonFor(index)}</span>{selectedGuide?.trial_interval && <span>Last computed trial: {selectedGuide.trial_interval.min} to {selectedGuide.trial_interval.max}. {selectedGuide.trial_interval.basis}</span>}</div>
    <div className="studio-measured-actions"><button disabled={disabled} onClick={() => reset(index)}>Reset {slots[index].label} to Default</button><button disabled={disabled} onClick={() => onEdit({ values: [...record.root.values], ab: {} })}>Reset all blocks to Default</button>{suggestedValues && <button disabled={disabled} onClick={() => onEdit({ values: [...suggestedValues], ab: {} })}>Restore last computed multipliers</button>}</div>
    <p className="studio-help">Edits become a personal draft. Save a named revision, then prepare again before copying. The original line measures parameter updates, not image influence.</p>
    <details><summary>Graph controls and measurement details</summary><p className="studio-help">Drag bars vertically; negative bars keep their sign. Numeric entries accept signed values. Arrows move/select or adjust by 0.01; Shift adjusts by 0.1; Delete or double-click resets a block. ↑ marks values above the fixed plot scale. Resets change block multipliers only; supporting model strength stays as shown.</p><p className="studio-help">Colours and trial intervals describe the last computed experiment; they are not proven safe ranges. This reference comes from measurement {reference.jobId} and remains a display reference during edits. Fresh preparation is required before Copy.</p></details>
  </div>;
}
