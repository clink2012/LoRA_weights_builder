import { useState } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import VersionPanel from "./VersionPanel";

const profile = { version_id: "default-one", default_id: "default-one", kind: "default", name: "Default", sequence: 1, values: [1, -0.25], binding: { slots: [{ label: "BASE" }, { label: "DOUBLE 0" }] }, settings: { role: "person", strength_model: 1 }, ab: { A: { slot_labels: ["BASE"], min: 0, max: 1, value: 1 } } };
const record = () => ({ selected: structuredClone(profile), versions: [structuredClone(profile)], name: "Keep my draft", busy: false });
const keyboardActivate = (button) => { button.focus(); fireEvent.click(button, { detail: 0 }); };
afterEach(cleanup);

function Harness({ opening = Promise.resolve(), initial, failed = false }) {
  const [current, setCurrent] = useState(initial);
  const actions = {
    open: async () => { setCurrent({ busy: true }); await opening; setCurrent(failed ? { error: "Source unavailable", busy: false } : record()); },
    editValue: (_, index, value) => setCurrent((previous) => { const draft = previous.draft || structuredClone(previous.selected); draft.values = [...draft.values]; draft.values[index] = value; return { ...previous, draft }; }),
    edit: (_, update) => setCurrent((previous) => ({ ...previous, draft: { ...(previous.draft || structuredClone(previous.selected)), ...update } })),
    setName: (_, name) => setCurrent((previous) => ({ ...previous, name })),
    save: vi.fn(), select: vi.fn(), discard: vi.fn(),
  };
  return <VersionPanel id="one" slotIndex={1} actions={actions} record={current} loading={false} />;
}

describe("version editor keyboard focus", () => {
  it("hands keyboard focus into the relevant exact editor after history opens", async () => {
    let finish; const opening = new Promise((resolve) => { finish = resolve; });
    render(<Harness opening={opening} />);
    keyboardActivate(screen.getByRole("button", { name: "Open variants & history" }));
    expect(screen.queryByLabelText("New exact value")).toBeNull();
    await act(async () => { finish(); });
    expect(document.activeElement).toBe(screen.getByLabelText("New exact value"));
    expect(document.activeElement.value).toBe("-0.25");
    expect(screen.getByText(/Editing DOUBLE 0/)).toBeTruthy();
  });
  it("restores focus after Apply without replacing the draft name or other settings", () => {
    const initial = record(); initial.draft = structuredClone(profile);
    render(<Harness initial={initial} />);
    fireEvent.change(screen.getByLabelText("New exact value"), { target: { value: "-0.375" } });
    keyboardActivate(screen.getByRole("button", { name: "Apply block value" }));
    expect(document.activeElement).toBe(screen.getByLabelText("New exact value"));
    expect(document.activeElement.value).toBe("-0.375");
    expect(screen.getByLabelText("Revision name").value).toBe("Keep my draft");
    expect(screen.getByText(/A · BASE · 0 to 1/)).toBeTruthy();
    expect(screen.getByLabelText("LoRA role").value).toBe("person");
    const name = screen.getByLabelText("Revision name"); name.focus();
    fireEvent.change(name, { target: { value: "Still my draft" } });
    expect(document.activeElement).toBe(name);
    expect(screen.getByLabelText("New exact value").value).toBe("-0.375");
  });
  it("also retains focus when equivalent numeric text does not remount the editor", () => {
    render(<Harness initial={record()} />);
    fireEvent.change(screen.getByLabelText("New exact value"), { target: { value: "-0.2500" } });
    keyboardActivate(screen.getByRole("button", { name: "Apply block value" }));
    expect(document.activeElement).toBe(screen.getByLabelText("New exact value"));
    expect(screen.getByText("Personal draft")).toBeTruthy();
  });
  it("restores the opening trigger after a failed open", async () => {
    render(<Harness failed />);
    keyboardActivate(screen.getByRole("button", { name: "Open variants & history" }));
    await screen.findByRole("alert");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Open variants & history" }));
  });
  it("restores the supporting strength editor after its Apply action", () => {
    render(<Harness initial={record()} />);
    fireEvent.click(screen.getByText("LoRA role and supporting settings"));
    fireEvent.change(screen.getByLabelText("Model strength"), { target: { value: "0.8123" } });
    keyboardActivate(screen.getByRole("button", { name: "Apply model strength" }));
    expect(document.activeElement).toBe(screen.getByLabelText("Model strength"));
    expect(document.activeElement.value).toBe("0.8123");
    expect(screen.getByLabelText("New exact value").value).toBe("-0.25");
  });
  it("does not send a pending opening's focus into a different LoRA", async () => {
    const actions = { open: vi.fn() };
    const { rerender } = render(<><button>Outside</button><VersionPanel id="one" slotIndex={1} actions={actions} /></>);
    keyboardActivate(screen.getByRole("button", { name: "Open variants & history" }));
    screen.getByRole("button", { name: "Outside" }).focus();
    rerender(<><button>Outside</button><VersionPanel id="two" slotIndex={1} actions={actions} record={record()} /></>);
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "Outside" })));
  });
});
