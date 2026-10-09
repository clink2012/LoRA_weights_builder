import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import MeasurementPanel from "./MeasurementPanel";

const items = [{ stable_id: "one", filename: "Portrait" }, { stable_id: "two", filename: "Clothing" }];
const entries = items.map((item) => ({ stable_id: item.stable_id, profile_version_id: `${item.stable_id}-default` }));
const props = { apiBase: "/api", selectedItems: items, versionIds: { one: "one-default", two: "two-default" }, result: { compatible: true, preparation_digest: "digest-one" }, dirty: false, loading: false };
const metrics = { status: "complete", measurement_basis: "effective_native_parameter_update", outer_model_strength_applied: false, block_weights_applied: false, slot_labels: ["BASE", ...Array.from({ length: 57 }, (_, index) => `BLOCK ${index}`)], sources: items.map((_, index) => ({ source_index: index, block_norms: Array(58).fill(index + 1), total_squared_norm: index + 1 })), pairs: [{ left_index: 0, right_index: 1, block_signed_cosines: [null, ...Array(57).fill(-0.25)] }] };
const job = (status = "complete", changes = {}) => ({ job_id: "job-one", status, target_contract_id: "flux1-dev-native-v1", entries, preparation_digest: "digest-one", metrics: status === "complete" ? metrics : null, ...changes });
const response = (data, ok = true, status = 200) => ({ ok, status, json: async () => data });
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("parameter measurements", () => {
  it("recovers matching saved measurements after fresh preparation without starting CPU work", async () => {
    const digest = "a".repeat(64), onJobChange = vi.fn();
    const fetch = vi.fn().mockResolvedValue(response({ status: "reused", job: job("complete", { preparation_digest: digest }) }));
    vi.stubGlobal("fetch", fetch);
    render(<MeasurementPanel {...props} result={{ compatible: true, preparation_digest: digest }} onJobChange={onJobChange} />);
    await screen.findByText("Saved measurements recovered and revalidated for the current composition.");
    expect(fetch.mock.calls[0][0]).toBe("/api/analysis-jobs/resolve");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ entries, target_contract_id: "flux1-dev-native-v1", expected_preparation_digest: digest });
    // The status renders before its passive notification effect necessarily
    // runs. Assert the delivered measurement, not scheduler timing under CI.
    await waitFor(() => expect(onJobChange.mock.calls.at(-1)?.[0]?.metrics).toEqual(metrics));
    expect(fetch).toHaveBeenCalledOnce();
  });
  it("a manual measurement wins over a late saved lookup", async () => {
    const digest = "a".repeat(64), late = deferred();
    const fetch = vi.fn().mockReturnValueOnce(late.promise).mockResolvedValue(response(job("complete", { preparation_digest: digest, job_id: "new-manual" })));
    vi.stubGlobal("fetch", fetch); const onJobChange = vi.fn();
    render(<MeasurementPanel {...props} result={{ compatible: true, preparation_digest: digest }} onJobChange={onJobChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    await screen.findByText("Measurements complete for the current saved composition.");
    late.resolve(response({ status: "reused", job: job("complete", { preparation_digest: digest, job_id: "old-saved" }) }));
    await waitFor(() => expect(onJobChange.mock.calls.at(-1)[0].job_id).toBe("new-manual"));
    expect(screen.queryByText("Saved measurements recovered and revalidated for the current composition.")).toBeNull();
  });
  it("requires saved versions and current preparation before measuring", () => {
    const { rerender } = render(<MeasurementPanel {...props} versionIds={{}} />);
    expect(screen.getByRole("button", { name: "Measure current sources" }).disabled).toBe(true);
    rerender(<MeasurementPanel {...props} dirty />);
    expect(screen.getByRole("button", { name: "Measure current sources" }).disabled).toBe(true);
    rerender(<MeasurementPanel {...props} />);
    expect(screen.getByRole("button", { name: "Measure current sources" }).disabled).toBe(false);
  });
  it("sends exact saved references and shows parameter measurements without applying values", async () => {
    const fetch = vi.fn().mockResolvedValue(response(job())); vi.stubGlobal("fetch", fetch);
    render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    fireEvent.click(await screen.findByText("Inspect numerical measurements"));
    expect(screen.getByRole("region", { name: "Per-block parameter measurements" })).toBeTruthy();
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ entries, target_contract_id: "flux1-dev-native-v1", expected_preparation_digest: "digest-one" });
    expect(screen.getByText("Not defined")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /apply|copy|save/i })).toBeNull();
    expect(document.body.textContent).toContain("exclude your block weights and model strength");
  });
  it("does not show late measurements after the selected versions or preparation change", async () => {
    const pending = deferred(); vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending.promise));
    const { rerender } = render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    rerender(<MeasurementPanel {...props} versionIds={{ ...props.versionIds, one: "one-new" }} result={null} />);
    pending.resolve(response(job()));
    expect(await screen.findByText("Measurements belong to an earlier composition.")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Per-block parameter measurements" })).toBeNull();
  });
  it("polls a running job and exposes explicit cancellation", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(job("queued"))).mockResolvedValueOnce(response(job("running"))).mockResolvedValueOnce(response(job("cancelled")));
    vi.stubGlobal("fetch", fetch); render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    expect(await screen.findByText("Queued for local analysis.")).toBeTruthy();
    expect(await screen.findByText("Reading tensors and measuring parameter updates…")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel measurement" }));
    expect(await screen.findByText("Measurement cancelled.")).toBeTruthy();
    expect(fetch.mock.calls.at(-1)[0]).toBe("/api/analysis-jobs/job-one/cancel");
    expect(screen.queryByRole("region", { name: "Per-block parameter measurements" })).toBeNull();
  });
  it("keeps incomplete, stale and mismatched results out of the comparison", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(job("stale", { reason: "Source changed." }))).mockResolvedValueOnce(response(job("complete", { preparation_digest: "another" })));
    vi.stubGlobal("fetch", fetch); render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    expect(await screen.findByText("Source changed.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "The measurement response does not match the requested composition.");
    expect(screen.queryByRole("region", { name: "Per-block parameter measurements" })).toBeNull();
  });
  it("reports unavailable runtime and rejects nonfinite measurements", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response({ detail: "Optional analysis runtime unavailable." }, false, 503)).mockResolvedValueOnce(response(job("complete", { metrics: { ...metrics, sources: [{ ...metrics.sources[0], total_squared_norm: Infinity }, metrics.sources[1]] } })));
    vi.stubGlobal("fetch", fetch); render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    expect(await screen.findByText("Optional analysis runtime unavailable.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("did not contain valid parameter measurements"));
    expect(screen.queryByRole("region", { name: "Per-block parameter measurements" })).toBeNull();
  });
  it("cancels a job whose start response arrives after leaving the workspace", async () => {
    const pending = deferred();
    const fetch = vi.fn().mockReturnValueOnce(pending.promise).mockResolvedValue(response(job("cancelled")));
    vi.stubGlobal("fetch", fetch);
    const { unmount } = render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    unmount(); pending.resolve(response(job("queued")));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
    expect(fetch.mock.calls[1][0]).toBe("/api/analysis-jobs/job-one/cancel");
  });
  it("allows status retry after a polling failure without exposing partial measurements", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(job("queued"))).mockRejectedValueOnce(new Error("Connection interrupted.")).mockResolvedValueOnce(response(job("complete")));
    vi.stubGlobal("fetch", fetch); render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    expect(await screen.findByRole("button", { name: "Check job status" })).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Per-block parameter measurements" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Check job status" }));
    fireEvent.click(await screen.findByText("Inspect numerical measurements"));
    expect(screen.getByRole("region", { name: "Per-block parameter measurements" })).toBeTruthy();
  });
  it("does not regress a terminal poll after a delayed running cancellation response", async () => {
    const cancellation = deferred();
    const fetch = vi.fn().mockImplementation((url) => url.endsWith("/cancel") ? cancellation.promise : Promise.resolve(response(url.endsWith("analysis-jobs") ? job("queued") : job("cancelled"))));
    vi.stubGlobal("fetch", fetch); render(<MeasurementPanel {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    fireEvent.click(await screen.findByRole("button", { name: "Cancel measurement" }));
    expect(await screen.findByText("Measurement cancelled.")).toBeTruthy();
    cancellation.resolve(response(job("running")));
    await waitFor(() => expect(screen.getByRole("button", { name: "Measure current sources" }).disabled).toBe(false));
    expect(screen.getByText("Measurement cancelled.")).toBeTruthy();
  });
  it("invalidates copy authority for current drift but not a late result for an older composition", async () => {
    const invalidate = vi.fn(); const stale = deferred();
    const fetch = vi.fn().mockReturnValueOnce(stale.promise).mockResolvedValueOnce(response(job("stale")));
    vi.stubGlobal("fetch", fetch);
    const { rerender } = render(<MeasurementPanel {...props} onInvalidatePrepared={invalidate} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    rerender(<MeasurementPanel {...props} result={{ ...props.result, preparation_digest: "new" }} onInvalidatePrepared={invalidate} />);
    stale.resolve(response(job("stale")));
    await screen.findByText("Measurement stale."); expect(invalidate).not.toHaveBeenCalled();
    rerender(<MeasurementPanel {...props} onInvalidatePrepared={invalidate} />);
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    await waitFor(() => expect(invalidate).toHaveBeenCalledOnce());
  });
});
