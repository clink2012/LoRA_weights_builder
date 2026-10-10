import { useState } from "react";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import BlockChart from "./BlockChart";
const labels = ["BASE", ...Array.from({ length: 19 }, (_, i) => `DOUBLE ${i}`), ...Array.from({ length: 38 }, (_, i) => `SINGLE ${i}`)];
const contract = { slots: labels.map((label, i) => ({ label, value: i === 2 ? -0.123456789 : 1 })) };
function Chart() { const [slot, setSlot] = useState(0); return <BlockChart contract={contract} selectedSlot={slot} onSelect={setSlot} />; }
afterEach(cleanup);
describe("continuous architecture chart", () => {
  it("keeps all 58 exact slots on one canvas with colour group boundaries and truthful negative magnitude", () => {
    const { container } = render(<Chart />);
    const chart = screen.getByRole("group", { name: "Individual block chart" });
    expect(within(chart).getAllByRole("button")).toHaveLength(58);
    expect(container.querySelectorAll(".studio-chart-track")).toHaveLength(1);
    expect(container.querySelectorAll('[data-group="double"].studio-chart-bar')).toHaveLength(19);
    expect(container.querySelectorAll('[data-group="single"].studio-chart-bar')).toHaveLength(38);
    const negative = screen.getByRole("button", { name: "DOUBLE 1: -0.123456789" });
    expect(negative.classList.contains("is-negative")).toBe(true);
    expect(negative.querySelector(".studio-chart-fill").style.height).toBe("12.3456789%");
    expect(screen.getByRole("button", { name: "BASE: 1" }).querySelector(".studio-chart-fill").style.height).toBe("100%");
  });
  it("uses one tab stop and arrow/Home/End selection across group boundaries", () => {
    render(<Chart />);
    const base = screen.getByRole("button", { name: "BASE: 1" }); base.focus();
    fireEvent.keyDown(base, { key: "ArrowRight" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "DOUBLE 0: 1" }));
    fireEvent.keyDown(document.activeElement, { key: "End" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "SINGLE 37: 1" }));
    expect(screen.getAllByRole("button").filter((button) => button.tabIndex === 0)).toHaveLength(1);
    fireEvent.keyDown(document.activeElement, { key: "Home" }); expect(document.activeElement).toBe(base);
  });
  it("edits bars by pointer position, preserves negative sign, and resets without changing the saved scale", () => {
    const changed = vi.fn();
    const props = { contract, selectedSlot: 2, onSelect: vi.fn(), onEditValue: changed, originalValues: contract.slots.map(() => 1), savedValues: contract.slots.map((slot) => slot.value) };
    const { container, rerender } = render(<BlockChart {...props} />);
    const bar = screen.getByRole("button", { name: "DOUBLE 1: -0.123456789" });
    container.querySelectorAll('.studio-chart-bar').forEach((element, i) => {
      element.getBoundingClientRect = () => ({ left: i * 50, right: i * 50 + 48, height: 100, bottom: 100 });
    });
    // jsdom needs a mouse-backed PointerEvent constructor for coordinates.
    vi.stubGlobal("PointerEvent", MouseEvent);
    const line = container.querySelector("polyline").getAttribute("points");
    fireEvent.pointerDown(bar, { button: 0, clientX: 124, clientY: 75 });
    expect(changed).toHaveBeenLastCalledWith(2, -.5);
    fireEvent.pointerMove(bar, { buttons: 1, clientX: 124, clientY: 25 });
    expect(changed).toHaveBeenLastCalledWith(2, -1.5);
    fireEvent.pointerUp(bar);
    fireEvent.keyDown(bar, { key: "ArrowUp", shiftKey: true });
    expect(changed).toHaveBeenLastCalledWith(2, Number((-.123456789 + .1).toPrecision(15)));
    fireEvent.doubleClick(bar);
    expect(changed).toHaveBeenLastCalledWith(2, 1);
    rerender(<BlockChart {...props} contract={{ slots: contract.slots.map((slot, index) => index === 2 ? { ...slot, value: -4 } : slot) }} />);
    expect(container.querySelector("polyline").getAttribute("points")).toBe(line);
    expect(screen.getByRole("button", { name: "DOUBLE 1: -4" }).querySelector(".studio-chart-fill").style.height).toBe("100%");
    vi.unstubAllGlobals();
  });
  it("blocks edits during profile operations and keeps exact signed numeric entries", () => {
    const changed = vi.fn();
    const props = { contract, selectedSlot: 0, onSelect: vi.fn(), onEditValue: changed };
    const { rerender } = render(<BlockChart {...props} disabled />);
    const bar = screen.getByRole("button", { name: "BASE: 1" });
    fireEvent.keyDown(bar, { key: "ArrowDown" });
    fireEvent.pointerDown(bar, { button: 0, clientY: 1 });
    expect(changed).not.toHaveBeenCalled();
    expect(screen.getByRole("textbox", { name: "Multiplier for BASE" }).disabled).toBe(true);
    rerender(<BlockChart {...props} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Multiplier for BASE" }), { target: { value: "-3.123456789" } });
    expect(changed).toHaveBeenLastCalledWith(0, -3.123456789);
  });
  it("draws through captured-pointer coordinates, fills skipped bars and stops on release or cancel", () => {
    const changed = vi.fn(), selected = vi.fn();
    const { container } = render(<BlockChart contract={contract} selectedSlot={0} onSelect={selected} onEditValue={changed} />);
    const bars = [...container.querySelectorAll('.studio-chart-bar')];
    bars.forEach((bar, i) => { bar.getBoundingClientRect = () => ({ left: i * 50, right: i * 50 + 48, bottom: 100, height: 100 }); });
    vi.stubGlobal('PointerEvent', MouseEvent);
    // Events stay on the first captured bar while x crosses positive/negative slots.
    fireEvent.pointerDown(bars[0], { button: 0, clientX: 24, clientY: 75 });
    fireEvent.pointerMove(bars[0], { buttons: 1, clientX: 174, clientY: 25 });
    expect(changed.mock.calls.map(([i]) => i)).toEqual([0, 1, 2, 3]);
    expect(changed.mock.calls[1][1]).toBeCloseTo(5 / 6);
    expect(changed.mock.calls[2][1]).toBeCloseTo(-7 / 6);
    expect(changed).toHaveBeenLastCalledWith(3, 1.5);
    fireEvent.pointerUp(bars[0]); fireEvent.click(bars[0]);
    expect(selected).toHaveBeenLastCalledWith(3);
    changed.mockClear(); fireEvent.pointerMove(bars[0], { buttons: 1, clientX: 224, clientY: 20 });
    expect(changed).not.toHaveBeenCalled();
    fireEvent.pointerDown(bars[3], { button: 0, clientX: 174, clientY: 25 });
    fireEvent.pointerMove(bars[3], { buttons: 1, clientX: 24, clientY: 75 });
    expect(changed.mock.calls.map(([i]) => i)).toEqual([3, 2, 1, 0]);
    fireEvent.pointerCancel(bars[3]); changed.mockClear();
    fireEvent.pointerMove(bars[3], { buttons: 1, clientX: 74, clientY: 50 });
    expect(changed).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
