import { useState } from "react";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
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
});
