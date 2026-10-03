import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useLibraryScan } from "./useLibraryScan";
const response = (data) => ({ ok: true, json: async () => data });
function Harness({ start, ready }) { const scan = useLibraryScan("/api", start, ready); return <><span>{scan.status?.phase || "Waiting"}</span><button disabled={scan.inventoryBusy}>Prepare</button><button onClick={scan.refresh}>Refresh</button><span>{scan.error}</span></>; }
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("background library lifecycle", () => {
  it("only reads startup status, gates inventory then permits editing during headers and notifies each inventory once", async () => {
    const start = vi.fn(), ready = vi.fn();
    const fetch = vi.fn().mockResolvedValueOnce(response({ job_id: "one", status: "running", phase: "catalogue" })).mockResolvedValue(response({ job_id: "one", status: "running", phase: "headers", catalogue_scan_id: "inventory-one", scanned: 2, total: 5 }));
    vi.stubGlobal("fetch", fetch); render(<Harness start={start} ready={ready} />);
    await screen.findByText("catalogue"); expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(true);
    expect(fetch.mock.calls[0][1]).toBeUndefined();
    await screen.findByText("headers", {}, { timeout: 2000 });
    expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(false);
    expect(start).toHaveBeenCalledTimes(1); expect(ready).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(fetch.mock.calls.length).toBeGreaterThan(2), { timeout: 2000 }); expect(ready).toHaveBeenCalledTimes(1);
  });
  it("ignores an older status read after manual refresh begins", async () => {
    let finish; const start = vi.fn(), ready = vi.fn();
    const fetch = vi.fn().mockReturnValueOnce(new Promise((resolve) => { finish = resolve; })).mockResolvedValue(response({ job_id: "new", status: "running", phase: "catalogue" }));
    vi.stubGlobal("fetch", fetch); render(<Harness start={start} ready={ready} />);
    fireEvent.click(screen.getByRole("button", { name: "Refresh" })); await screen.findByText("catalogue");
    await act(async () => finish(response({ status: "complete", phase: "finished", catalogue_scan_id: "old" })));
    expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(true); expect(ready).not.toHaveBeenCalled();
    expect(fetch.mock.calls[1][1]).toMatchObject({ method: "POST", body: "{}" });
  });
});
