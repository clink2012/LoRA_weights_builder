import { useState } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import MeasuredBlockChart from "./MeasuredBlockChart";

const labels = ["BASE", "DOUBLE 0", "SINGLE 0"];
const initial = { version_id: "personal", default_id: "default", values: [1, .5, -.25], settings: { strength_model: .8 }, binding: { slots: labels.map((label) => ({ label })) }, ab: { A: { slot_labels: ["DOUBLE 0"], value: .5 } } };
const root = { ...initial, version_id: "default", values: [1, 1, 0] };
const reference = { norms: [0, 12, 4], jobId: "job-original" };
const guidance = [{ state: "inactive", reason: "Zero" }, { state: "suggested", reason: "Measured pressure trial", trial_interval: { min: .4, max: .5, basis: "Trial only" } }, { state: "protected", reason: "Protected role" }];
function Chart({ disabled = false, changed, originalReference = reference, modelStrength = .8, suggestedValues }) {
  const [draft, setDraft] = useState(null); const [slot, setSlot] = useState(1);
  const selected = { ...initial, settings: { strength_model: modelStrength } };
  return <MeasuredBlockChart record={{ selected, root, draft }} reference={originalReference} guidance={guidance} suggestedValues={suggestedValues} selectedSlot={slot} onSelect={setSlot} disabled={disabled} onEditValue={(index, value) => { changed?.(index, value); setDraft((previous) => { const values = [...(previous || selected).values]; values[index] = value; return { ...(previous || selected), values }; }); }} onEdit={(update) => setDraft((previous) => ({ ...(previous || selected), ...update }))} />;
}
afterEach(cleanup);

describe("Measured original and editable contributions", () => {
  it("shows 12 original versus 4.8 current while keeping the multiplier at 0.5", () => {
    const { container } = render(<Chart />);
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("0.5");
    const bar = screen.getByRole("button", { name: /DOUBLE 0: multiplier 0.5, current update 4.8.*original update 12/ });
    expect(parseFloat(bar.querySelector(".studio-measured-fill").style.height)).toBeCloseTo(40);
    expect(container.querySelector(".studio-original-line polyline").getAttribute("points")).toBe("25,100 75,0 125,66.66666666666667");
    expect(bar.dataset.guidance).toBe("suggested");
    expect(screen.getByText(/Last computed trial: 0.4 to 0.5/)).toBeTruthy();
  });

  it("edits exact signed multipliers without changing the original line or display peak", () => {
    const { container } = render(<Chart />); const line = container.querySelector("polyline").getAttribute("points");
    fireEvent.change(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }), { target: { value: "-2.123456789" } });
    const bar = screen.getByRole("button", { name: /DOUBLE 0: multiplier -2.123456789/ });
    expect(bar.classList.contains("is-negative")).toBe(true);
    expect(bar.querySelector(".studio-measured-fill").style.height).toBe("100%");
    expect(bar.querySelector(".studio-measured-overflow").textContent).toBe("↑");
    expect(container.querySelector("polyline").getAttribute("points")).toBe(line);
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("-2.123456789");
  });

  it("allows typing the minus sign as an intermediate state and rejects non-finite text", () => {
    const changed = vi.fn(); render(<Chart changed={changed} />);
    const input = screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" });
    fireEvent.change(input, { target: { value: "-" } });
    expect(input.value).toBe("-"); expect(changed).not.toHaveBeenCalled();
    fireEvent.change(input, { target: { value: "-0.23456789" } });
    expect(changed).toHaveBeenLastCalledWith(1, -.23456789);
    fireEvent.change(input, { target: { value: "Infinity" } }); fireEvent.blur(input);
    expect(input.value).toBe("-0.23456789"); expect(input.getAttribute("aria-invalid")).toBe("true");
  });

  it("resets a single block or every block to the immutable Default", () => {
    render(<Chart />);
    fireEvent.click(screen.getByRole("button", { name: "Reset DOUBLE 0 to Default" }));
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("1");
    expect(screen.getByRole("textbox", { name: "Multiplier for SINGLE 0" }).value).toBe("-0.25");
    fireEvent.click(screen.getByRole("button", { name: "Reset all blocks to Default" }));
    expect(screen.getByRole("textbox", { name: "Multiplier for SINGLE 0" }).value).toBe("0");
    expect(screen.getByText(/Model strength: 0.8/)).toBeTruthy();
  });

  it("supports keyboard selection, signed adjustment and individual reset", () => {
    render(<Chart />); const bar = screen.getByRole("button", { name: /DOUBLE 0: multiplier 0.5/ });
    fireEvent.keyDown(bar, { key: "ArrowUp", shiftKey: true });
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("0.6");
    fireEvent.keyDown(bar, { key: "ArrowRight" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /SINGLE 0: multiplier -0.25/ }));
    fireEvent.keyDown(document.activeElement, { key: "Delete" });
    expect(screen.getByRole("textbox", { name: "Multiplier for SINGLE 0" }).value).toBe("0");
  });

  it("restores computed multipliers separately from original Default values", () => {
    render(<Chart suggestedValues={[1, .45, -.2]} />);
    fireEvent.click(screen.getByRole("button", { name: "Restore last computed multipliers" }));
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("0.45");
    fireEvent.click(screen.getByRole("button", { name: "Reset all blocks to Default" }));
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).value).toBe("1");
  });

  it("inverts the contribution scale for a drag while preserving multiplier sign", () => {
    const changed = vi.fn(); const { container } = render(<Chart changed={changed} />);
    const bar = screen.getByRole("button", { name: /SINGLE 0: multiplier -0.25/ });
    bar.getBoundingClientRect = () => ({ bottom: 200, height: 200 });
    fireEvent.pointerDown(bar, { button: 0, pointerId: 1, clientY: 100 });
    fireEvent.pointerUp(bar, { pointerId: 1 });
    // 50% of peak12 / (strength0.8 * norm4) = -1.875.
    expect(changed).toHaveBeenCalledWith(2, -1.875);
    expect(container.querySelector(".studio-original-line")).toBeTruthy();
  });

  it("keeps zero-update blocks finite and never manufactures measured content", () => {
    const changed = vi.fn(); render(<Chart changed={changed} originalReference={{ norms: [0, 0, 0], jobId: "zero" }} />);
    fireEvent.change(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }), { target: { value: "2" } });
    const bar = screen.getByRole("button", { name: /DOUBLE 0: multiplier 2, current update 0, original update 0/ });
    expect(bar.dataset.guidance).toBe("inactive"); expect(bar.querySelector(".studio-measured-fill").style.height).toBe("0%");
    bar.getBoundingClientRect = () => ({ bottom: 200, height: 200 }); changed.mockClear();
    fireEvent.pointerDown(bar, { button: 0, pointerId: 1, clientY: 100 });
    expect(changed).not.toHaveBeenCalled();
  });

  it("disables edits and resets during a pending operation", () => {
    render(<Chart disabled />);
    expect(screen.getByRole("textbox", { name: "Multiplier for DOUBLE 0" }).disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Reset all blocks to Default" }).disabled).toBe(true);
  });
});
