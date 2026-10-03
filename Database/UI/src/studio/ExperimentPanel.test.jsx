import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ExperimentPanel from "./ExperimentPanel";

const entries = [{ stable_id: "one", profile_version_id: "one-default" }, { stable_id: "two", profile_version_id: "two-default" }];
const before = Array(58).fill(1);
const after = before.map((value, index) => index === 1 ? 0.8 : value);
const props = { apiBase: "/api", job: { job_id: "job-one", status: "complete", entries, preparation_digest: "digest" }, selectedItems: [{ stable_id: "one", filename: "Portrait" }, { stable_id: "two", filename: "Clothing" }], versionIds: { one: "one-default", two: "two-default" }, result: { preparation_digest: "digest" }, dirty: false, loading: false, selectedId: "one", selectedSlot: 1, onBusyChange: vi.fn(), onRestore: vi.fn(), onInvalidatePrepared: vi.fn() };
const preview = (changed = true) => ({ job_id: "job-one", input_preparation_digest: "digest", proposal_digest: "proposal", can_save: changed, ab_handling: "Earlier A/B settings remain in parent versions.", policy_preview: { status: "experimental_preview", calibrated: false, constants: { max_reduction: 0.2, pressure_threshold: 0.1 }, entries: entries.map((entry, index) => ({ ...entry, before_values: before, values: changed && index === 0 ? after : before })), changes: changed ? [{ stable_id: "one", slot_index: 1, slot_label: "DOUBLE 0", before: 1, value: 0.8, min: 0.8, max: 1, reason: "Positive parameter alignment", basis: "Uncalibrated trial" }] : [], blocks: [], limitations: [] } });
const response = (data, ok = true, status = 200) => ({ ok, status, json: async () => data });
const saved = { status: "saved", composition: { version_id: "recipe-one", entries, historical_snapshot: { unsafe: "never copy" } } };
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });
async function propose() { fireEvent.click(screen.getByRole("button", { name: "Preview block experiment" })); await screen.findByRole("heading", { name: /block changes proposed/ }); }

describe("guided experiments", () => {
  it("uses explicit Normal priorities and disables saving a no-change preview", async () => {
    const fetch = vi.fn().mockResolvedValue(response(preview(false))); vi.stubGlobal("fetch", fetch);
    render(<ExperimentPanel {...props} />); await propose();
    expect(JSON.parse(fetch.mock.calls[0][1].body).priorities).toEqual({ one: 1, two: 1 });
    expect(screen.getByRole("button", { name: "Save as new experiment" }).disabled).toBe(true);
    expect(document.body.textContent).toContain("does not prove that the LoRAs are compatible");
  });
  it("reviews changes and sends server digests rather than client vectors to save", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(preview())).mockResolvedValueOnce(response(saved)); vi.stubGlobal("fetch", fetch);
    render(<ExperimentPanel {...props} />);
    fireEvent.change(screen.getByLabelText("Priority for Portrait"), { target: { value: "0" } }); await propose();
    expect(screen.getByText("1 → 0.8", { selector: "strong" })).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Experiment name"), { target: { value: "Gentle trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Save as new experiment" }));
    await waitFor(() => expect(props.onRestore).toHaveBeenCalledWith(saved.composition));
    const body = JSON.parse(fetch.mock.calls[1][1].body);
    expect(body).toMatchObject({ job_id: "job-one", priorities: { one: 0, two: 1 }, expected_preparation_digest: "digest", expected_proposal_digest: "proposal", name: "Gentle trial" });
    expect(body.values).toBeUndefined(); expect(body.idempotency_key).toBeTruthy();
  });
  it("invalidates a preview when priorities, versions or preparation change", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(preview())));
    const { rerender } = render(<ExperimentPanel {...props} />); await propose();
    fireEvent.change(screen.getByLabelText("Priority for Clothing"), { target: { value: "2" } });
    expect(screen.queryByRole("button", { name: "Save as new experiment" })).toBeNull();
    await propose(); rerender(<ExperimentPanel {...props} versionIds={{ ...props.versionIds, one: "new-version" }} result={null} />);
    expect(screen.queryByRole("button", { name: "Save as new experiment" })).toBeNull();
    expect(screen.getByRole("button", { name: "Preview block experiment" }).disabled).toBe(true);
  });
  it("locks editing during pending requests and ignores late responses after workspace removal", async () => {
    const pending = deferred(); vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending.promise));
    const { unmount } = render(<ExperimentPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Preview block experiment" }));
    expect(props.onBusyChange).toHaveBeenCalledWith(true);
    expect(screen.getByLabelText("Priority for Portrait").disabled).toBe(true);
    unmount(); pending.resolve(response(preview()));
    await pending.promise;
    expect(props.onRestore).not.toHaveBeenCalled();
  });
  it("reuses one idempotency key only for an exact retry after uncertain save", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(preview())).mockRejectedValueOnce(new Error("connection")).mockResolvedValueOnce(response(saved)); vi.stubGlobal("fetch", fetch);
    render(<ExperimentPanel {...props} />); await propose();
    fireEvent.change(screen.getByLabelText("Experiment name"), { target: { value: "Trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Save as new experiment" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", expect.stringContaining("save outcome may be unknown"));
    fireEvent.click(screen.getByRole("button", { name: "Save as new experiment" }));
    await waitFor(() => expect(props.onRestore).toHaveBeenCalled());
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual(JSON.parse(fetch.mock.calls[2][1].body));
  });
  it("invalidates prepared copy authority on a stale save conflict", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(response(preview())).mockResolvedValueOnce(response({ detail: "Source changed." }, false, 409)));
    render(<ExperimentPanel {...props} />); await propose();
    fireEvent.change(screen.getByLabelText("Experiment name"), { target: { value: "Trial" } });
    fireEvent.click(screen.getByRole("button", { name: "Save as new experiment" }));
    expect(await screen.findByText("Source changed.")).toBeTruthy();
    expect(props.onInvalidatePrepared).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Save as new experiment" })).toBeNull();
  });
  it("invalidates prepared copy authority when preview detects source drift", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ detail: "Measured sources changed." }, false, 409)));
    render(<ExperimentPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Preview block experiment" }));
    expect(await screen.findByText("Measured sources changed.")).toBeTruthy();
    expect(props.onInvalidatePrepared).toHaveBeenCalledOnce();
  });
  it("does not resurrect a previous preview after a draft is discarded", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(preview())));
    const { rerender } = render(<ExperimentPanel {...props} />); await propose();
    rerender(<ExperimentPanel {...props} dirty />);
    rerender(<ExperimentPanel {...props} />);
    expect(screen.queryByRole("button", { name: "Save as new experiment" })).toBeNull();
  });
});
