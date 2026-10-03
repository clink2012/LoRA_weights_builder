import { useRef } from "react";

const groupOf = (slot) => slot.label === "BASE" ? "base" : slot.label.startsWith("DOUBLE") ? "double" : slot.label.startsWith("SINGLE") ? "single" : "other";
const groupName = (group) => ({ base: "BASE", double: "Double blocks", single: "Single blocks", other: "Other blocks" })[group];

export default function BlockChart({ contract, selectedSlot, onSelect }) {
  const buttons = useRef([]);
  const slots = contract.slots;
  const extent = Math.max(1, ...slots.map((slot) => Math.abs(slot.value)));
  const current = Math.min(selectedSlot, slots.length - 1);
  const groups = [];
  slots.forEach((slot, index) => {
    const group = groupOf(slot);
    if (groups.at(-1)?.group === group) groups.at(-1).count += 1;
    else groups.push({ group, start: index, count: 1 });
  });
  const columns = slots.length > 1 && slots[0]?.label === "BASE" ? `minmax(32px, 2fr) repeat(${slots.length - 1}, minmax(0, 1fr))` : `repeat(${slots.length}, minmax(0, 1fr))`;
  function move(event, index) {
    const next = event.key === "ArrowRight" ? Math.min(slots.length - 1, index + 1) : event.key === "ArrowLeft" ? Math.max(0, index - 1) : event.key === "Home" ? 0 : event.key === "End" ? slots.length - 1 : null;
    if (next === null) return;
    event.preventDefault(); onSelect(next); buttons.current[next]?.focus();
  }
  return <div className="studio-unified-chart">
    <div className="studio-chart-legend">{groups.map(({ group, start, count }) => <span key={start} data-group={group}><i aria-hidden="true" />{groupName(group)}<small>{count}</small></span>)}</div>
    <div className="studio-chart-scroll" role="group" aria-label="Individual block chart" aria-describedby="studio-chart-keyboard-help">
      <div className="studio-chart-canvas" style={{ "--slot-count": slots.length, "--chart-columns": columns }}>
        <div className="studio-chart-track">{slots.map((slot, index) => <button ref={(element) => { buttons.current[index] = element; }} key={slot.label} type="button" data-group={groupOf(slot)} className={`studio-chart-bar ${index > 0 && groupOf(slots[index - 1]) !== groupOf(slot) ? "is-group-start" : ""} ${current === index ? "is-current" : ""} ${slot.value < 0 ? "is-negative" : ""}`} aria-label={`${slot.label}: ${slot.value}`} aria-pressed={current === index} tabIndex={current === index ? 0 : -1} onKeyDown={(event) => move(event, index)} onClick={() => onSelect(index)} title={`${slot.label}: ${slot.value}`}>
          <span className="studio-chart-fill" style={{ height: `${Math.abs(slot.value) / extent * 100}%` }} />
          {slot.value < 0 && <span className="studio-chart-sign" aria-hidden="true">−</span>}
          <span className="studio-chart-selection" aria-hidden="true">{current === index ? "●" : ""}</span>
        </button>)}</div>
        <div className="studio-chart-group-labels">{groups.map(({ group, start, count }) => <span key={start} style={{ gridColumn: `${start + 1} / span ${count}` }}><strong>{group === "base" ? "BASE" : `${group === "double" ? "Double" : group === "single" ? "Single" : "Other"} · ${count}`}</strong>{count > 1 && <small>{slots[start].label.split(" ").at(-1)}–{slots[start + count - 1].label.split(" ").at(-1)}</small>}</span>)}</div>
      </div>
    </div>
    <p id="studio-chart-keyboard-help" className="studio-chart-help">{slots.length} exact slots · arrows move between blocks · Home / End jump to the edges. Heights show magnitude; striped bars and − mark negative values.</p>
  </div>;
}
