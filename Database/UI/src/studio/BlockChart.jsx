import { useRef } from "react";
import { MultiplierInput } from "./MeasuredBlockChart";

const groupOf = (slot) => slot.label === "BASE" ? "base" : slot.label.startsWith("DOUBLE") ? "double" : slot.label.startsWith("SINGLE") ? "single" : "other";
const groupName = (group) => ({ base: "BASE", double: "Double blocks", single: "Single blocks", other: "Other blocks" })[group];

export default function BlockChart({ contract, selectedSlot, onSelect, onEditValue, originalValues, savedValues, disabled = false }) {
  const buttons = useRef([]);
  const dragging = useRef(null);
  const slots = contract.slots;
  const editable = Boolean(onEditValue);
  // Keep headroom and a stable scale for drafts; numbers can exceed the display.
  const extent = Math.max(editable ? 2 : 1, ...(savedValues || slots.map((slot) => slot.value)).map(Math.abs), ...(originalValues || []).map(Math.abs));
  const current = Math.min(selectedSlot, slots.length - 1);
  const groups = [];
  slots.forEach((slot, index) => {
    const group = groupOf(slot);
    if (groups.at(-1)?.group === group) groups.at(-1).count += 1;
    else groups.push({ group, start: index, count: 1 });
  });
  const columns = editable ? `repeat(${slots.length}, minmax(48px, 1fr))` : slots.length > 1 && slots[0]?.label === "BASE" ? `minmax(32px, 2fr) repeat(${slots.length - 1}, minmax(0, 1fr))` : `repeat(${slots.length}, minmax(0, 1fr))`;
  function move(event, index) {
    const next = event.key === "ArrowRight" ? Math.min(slots.length - 1, index + 1) : event.key === "ArrowLeft" ? Math.max(0, index - 1) : event.key === "Home" ? 0 : event.key === "End" ? slots.length - 1 : null;
    if (next !== null) {
      event.preventDefault(); onSelect(next); buttons.current[next]?.focus({ preventScroll: true }); buttons.current[next]?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    } else if (editable && !disabled && ["ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      const value = slots[index].value + (event.key === "ArrowUp" ? 1 : -1) * (event.shiftKey ? .1 : .01);
      if (Number.isFinite(value)) onEditValue(index, Number(value.toPrecision(15)));
    } else if (editable && !disabled && event.key === "Delete" && originalValues) {
      event.preventDefault(); onEditValue(index, originalValues[index]);
    }
  }
  function drag(event, index) {
    if (!editable || disabled) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.height) return;
    const ratio = Math.max(0, Math.min(1, (rect.bottom - event.clientY) / rect.height));
    onEditValue(index, (dragging.current?.sign || (slots[index].value < 0 ? -1 : 1)) * ratio * extent);
  }
  return <div className="studio-unified-chart">
    <div className="studio-chart-legend">{groups.map(({ group, start, count }) => <span key={start} data-group={group}><i aria-hidden="true" />{groupName(group)}<small>{count}</small></span>)}</div>
    <div className="studio-chart-scroll" role="group" aria-label="Individual block chart" aria-describedby="studio-chart-keyboard-help">
      <div className={`studio-chart-canvas ${editable ? "is-editable" : ""}`} style={{ "--slot-count": slots.length, "--chart-columns": columns }}>
        {editable && <div className="studio-block-values">{slots.map((slot, index) => <MultiplierInput key={slot.label} label={slot.label} value={slot.value} disabled={disabled} onSelect={() => onSelect(index)} onChange={(value) => onEditValue(index, value)} />)}</div>}
        <div className="studio-chart-plot">
        {originalValues && <svg className="studio-original-line" viewBox={`0 0 ${slots.length * 50} 100`} preserveAspectRatio="none" aria-hidden="true"><polyline fill="none" points={originalValues.map((value, index) => `${index * 50 + 25},${100 - Math.min(100, Math.abs(value) / extent * 100)}`).join(" ")} /></svg>}
        <div className="studio-chart-track">{slots.map((slot, index) => <button ref={(element) => { buttons.current[index] = element; }} key={slot.label} type="button" disabled={disabled} data-group={groupOf(slot)} className={`studio-chart-bar ${index > 0 && groupOf(slots[index - 1]) !== groupOf(slot) ? "is-group-start" : ""} ${current === index ? "is-current" : ""} ${slot.value < 0 ? "is-negative" : ""}`} aria-label={`${slot.label}: ${slot.value}`} aria-pressed={current === index} tabIndex={current === index ? 0 : -1} onKeyDown={(event) => move(event, index)} onClick={() => onSelect(index)} title={`${slot.label}: ${slot.value}`} onDoubleClick={() => { if (editable && originalValues && !disabled) onEditValue(index, originalValues[index]); }} onPointerDown={(event) => { if (event.button !== 0 || disabled || !editable) return; onSelect(index); dragging.current = { index, sign: slot.value < 0 ? -1 : 1 }; event.currentTarget.setPointerCapture?.(event.pointerId); drag(event, index); }} onPointerMove={(event) => { if (dragging.current?.index === index) drag(event, index); }} onPointerUp={() => { dragging.current = null; }} onPointerCancel={() => { dragging.current = null; }}>
          <span className="studio-chart-fill" style={{ height: `${Math.min(100, Math.abs(slot.value) / extent * 100)}%` }} />
          {slot.value < 0 && <span className="studio-chart-sign" aria-hidden="true">−</span>}
          {Math.abs(slot.value) > extent && <span className="studio-chart-sign" title="Above the fixed display scale">↑</span>}
          <span className="studio-chart-selection" aria-hidden="true">{current === index ? "●" : ""}</span>
        </button>)}</div>
        </div>
        <div className="studio-chart-group-labels">{groups.map(({ group, start, count }) => <span key={start} style={{ gridColumn: `${start + 1} / span ${count}` }}><strong>{group === "base" ? "BASE" : `${group === "double" ? "Double" : group === "single" ? "Single" : "Other"} · ${count}`}</strong>{count > 1 && <small>{slots[start].label.split(" ").at(-1)}–{slots[start + count - 1].label.split(" ").at(-1)}</small>}</span>)}</div>
      </div>
    </div>
    <p id="studio-chart-keyboard-help" className="studio-chart-help">{slots.length} exact slots · arrows move between blocks · Home / End jump to the edges. Heights show magnitude; striped bars and − mark negative values. {editable && "Click or drag a bar to edit. Up / Down adjusts by 0.01; Shift by 0.1. Signed numbers appear above. Delete or double-click resets a captured Default block."}</p>
  </div>;
}
