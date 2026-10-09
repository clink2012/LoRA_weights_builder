import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useLibraryScan } from "./useLibraryScan";
const response = (data) => ({ ok: true, json: async () => data });
const fresh = response({ status: "current", checked_at: "2026-10-09T12:00:00Z", counts: { added: 0, removed: 0, changed: 0, returned: 0 } });
function Harness({ start, ready }) { const scan = useLibraryScan("/api", start, ready); return <><span>{scan.status?.phase || "Waiting"}</span><button disabled={scan.inventoryBusy}>Prepare</button><button onClick={scan.refresh}>Refresh</button><span>{scan.error}</span></>; }
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("background library lifecycle", () => {
  it("only reads startup status, gates inventory then permits editing during headers and notifies each inventory once", async () => {
    const start = vi.fn(), ready = vi.fn();
    let reads = 0;
    const fetch = vi.fn(async (url) => url.endsWith("/freshness") ? fresh : response(++reads === 1 ? { job_id: "one", status: "running", phase: "catalogue" } : { job_id: "one", status: "running", phase: "headers", catalogue_scan_id: "inventory-one", scanned: 2, total: 5 }));
    vi.stubGlobal("fetch", fetch); render(<Harness start={start} ready={ready} />);
    await screen.findByText("catalogue"); expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(true);
    expect(fetch.mock.calls[0][1]).toBeUndefined();
    await screen.findByText("headers", {}, { timeout: 2000 });
    expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(false);
    expect(start).toHaveBeenCalledTimes(1); expect(ready).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(reads).toBeGreaterThan(2), { timeout: 2000 }); expect(ready).toHaveBeenCalledTimes(1);
  });
  it("ignores an older status read after manual refresh begins", async () => {
    let finish; const start = vi.fn(), ready = vi.fn();
    const fetch = vi.fn((url, init) => url.endsWith("/freshness") ? Promise.resolve(fresh) : init?.method ? Promise.resolve(response({ job_id: "new", status: "running", phase: "catalogue" })) : new Promise((resolve) => { finish = resolve; }));
    vi.stubGlobal("fetch", fetch); render(<Harness start={start} ready={ready} />);
    fireEvent.click(screen.getByRole("button", { name: "Refresh" })); await screen.findByText("catalogue");
    await act(async () => finish(response({ status: "complete", phase: "finished", catalogue_scan_id: "old" })));
    expect(screen.getByRole("button", { name: "Prepare" }).disabled).toBe(true); expect(ready).not.toHaveBeenCalled();
    expect(fetch.mock.calls.find(([, init]) => init?.method)[1]).toMatchObject({ method: "POST", body: "{}" });
  });
});
