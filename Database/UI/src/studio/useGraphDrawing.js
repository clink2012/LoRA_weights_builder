import { useLayoutEffect, useRef } from "react";

// Pointer capture keeps a stroke alive, but coordinates choose the edited bar.
// Snapshot the scale and signs at press time so edits cannot rescale a stroke.
export default function useGraphDrawing({ buttons, values, valueAt, onSelect, onEditValue, disabled }) {
  const stroke = useRef(null);
  const suppressClick = useRef(false);
  useLayoutEffect(() => {
    const active = stroke.current;
    if (!active?.scrollParent) return;
    // Invalidating Copy can remove a result summary above the chart. Keep the
    // pressed bar under the pointer until release, before the browser paints.
    const top = buttons.current[active.start]?.getBoundingClientRect().top;
    if (Number.isFinite(top)) active.scrollParent.scrollTop += top - active.top;
  });
  function apply(index, y, active) {
    const rect = buttons.current[index]?.getBoundingClientRect();
    if (!rect?.height) return;
    const ratio = Math.max(0, Math.min(1, (rect.bottom - y) / rect.height));
    const value = active.valueAt(index, ratio, active.signs[index]);
    if (Number.isFinite(value)) onEditValue(index, value);
  }
  function finish(event) {
    if (!stroke.current || stroke.current.pointerId !== event.pointerId) return;
    suppressClick.current = stroke.current.moved;
    stroke.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  function move(event) {
    const active = stroke.current;
    if (!active || active.pointerId !== event.pointerId) return;
    if (disabled) { finish(event); return; }
    if (event.buttons === 0) { finish(event); return; }
    const rects = buttons.current.map((button) => button?.getBoundingClientRect());
    let index = active.last?.index ?? active.start;
    // Coordinate-less test environments still support vertical editing.
    if (Number.isFinite(rects[0]?.left)) {
      if (event.clientX < rects[0].left || event.clientX > rects.at(-1).right) { active.last = null; return; }
      index = rects.findIndex((rect, i) => event.clientX <= (i === rects.length - 1 ? rect.right : (rect.right + rects[i + 1].left) / 2));
    }
    if (index < 0) return;
    event.preventDefault();
    const previous = active.last;
    if (previous && previous.index !== index) {
      active.moved = true;
      const step = index > previous.index ? 1 : -1;
      for (let i = previous.index + step; i !== index; i += step) {
        const centre = (rects[i].left + rects[i].right) / 2;
        const t = Math.max(0, Math.min(1, (centre - previous.x) / (event.clientX - previous.x)));
        apply(i, previous.y + t * (event.clientY - previous.y), active);
      }
    }
    apply(index, event.clientY, active);
    onSelect(index);
    active.last = { index, x: event.clientX, y: event.clientY };
  }
  return (index) => ({
    onPointerDown(event) {
      if (event.button !== 0 || disabled || stroke.current) return;
      event.preventDefault();
      suppressClick.current = false;
      const active = { start: index, pointerId: event.pointerId, valueAt, signs: values.map((value) => value < 0 ? -1 : 1), moved: false, last: { index, x: event.clientX, y: event.clientY } };
      active.top = event.currentTarget.getBoundingClientRect().top;
      let ancestor = event.currentTarget.parentElement;
      while (ancestor && !(ancestor.scrollHeight > ancestor.clientHeight && /(auto|scroll)/.test(getComputedStyle(ancestor).overflowY))) ancestor = ancestor.parentElement;
      active.scrollParent = ancestor || document.scrollingElement;
      stroke.current = active;
      event.currentTarget.setPointerCapture?.(event.pointerId);
      onSelect(index);
      apply(index, event.clientY, active);
    },
    onPointerMove: move,
    onPointerUp: finish,
    onPointerCancel: finish,
    onLostPointerCapture: finish,
    onClick() { if (!suppressClick.current) onSelect(index); suppressClick.current = false; },
  });
}
