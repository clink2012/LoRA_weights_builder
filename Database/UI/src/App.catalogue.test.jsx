import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const item = (id, presence) => ({ id, stable_id: `sid-${id}`, filename: `${presence}-${id}.safetensors`, base_model_code: "FLX", category_code: "PPL", role: "person", role_source: "folder_hint", presence, architecture_verified: false });
describe("native library catalogue", () => {
  let refreshed, failRefresh, pendingCurrent, resumed;
  beforeEach(() => {
    refreshed = false; failRefresh = false; pendingCurrent = null; resumed = false;
    vi.stubGlobal("localStorage", { getItem: () => null, setItem: () => {} });
    vi.stubGlobal("fetch", vi.fn(async (input, init) => {
      const url = new URL(String(input), "http://localhost");
      if (url.pathname.endsWith("/model-families")) return response({ families: [{ code: "FLX", display_name: "FLUX.1" }] });
      if (url.pathname.endsWith("/composition-versions")) return response({ versions: [] });
      if (url.pathname.endsWith("/library-scan") && !init?.method) return response(resumed ? { status: "complete", phase: "finished", catalogue_scan_id: "earlier-inventory", catalogue: null } : { status: "idle" });
      if (url.pathname.endsWith("/library-scan")) { if (failRefresh) return response({ detail: { reason: "The local folder is unavailable; no changes were applied." } }, 503); refreshed = true; return response({ status: "complete", phase: "finished", catalogue_scan_id: "scan-1", catalogue: { counts: { present: 1, added: 1, missing: 1 } } }); }
      if (url.pathname.endsWith("/catalogue/compatible")) return response({ results: [item(1, "current")], total: 1, catalogue_status: "refreshed", counts: { eligible: 1, excluded: 0, unknown: 0 } });
      if (url.pathname.endsWith("/catalogue")) {
        const presence = url.searchParams.get("presence");
        if (presence === "current" && pendingCurrent) return pendingCurrent;
        const results = !refreshed && presence === "current" ? [] : [item(presence === "missing" ? 2 : 1, presence === "all" ? "unchecked" : presence)];
        return response({ results, total: results.length, catalogue_status: refreshed ? "refreshed" : "not_refreshed" });
      }
      if (url.pathname.endsWith("/lora/search")) return response({ results: [item(3, "missing")], total: 1 });
      if (init?.method === "POST") throw new Error(`Unexpected mutation ${url.pathname}`);
      return response({}, 404);
    }));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
  it("starts with current files, explains the first refresh and never scans on page load", async () => {
    render(<App />);
    expect(await screen.findByText(/Choose “Refresh library” to check current files/)).toBeTruthy();
    expect(screen.getByLabelText("Library view").value).toBe("current");
    expect(globalThis.fetch.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
    expect(screen.queryByRole("button", { name: /Rescan/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Refresh library" }));
    expect(await screen.findByRole("button", { name: /current-1.*sid-1/ })).toBeTruthy();
    const call = globalThis.fetch.mock.calls.find(([url, init]) => String(url).endsWith("/library-scan") && init?.method === "POST");
    expect(JSON.parse(call[1].body)).toEqual({});
    expect(globalThis.fetch.mock.calls.some(([url]) => /reindex_all|index_status/.test(String(url)))).toBe(false);
  });
  it("does not invent zero inventory totals when reopening a resumed header check", async () => {
    resumed = true; refreshed = true; render(<App />);
    await screen.findByText("Saved inventory loaded. Header checks use that inventory; saved history is preserved.");
    expect(document.body.textContent).not.toContain("0 current · 0 added · 0 missing");
  });
  it("offers missing history explicitly while keeping the selected stack", async () => {
    refreshed = true; render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /current-1.*sid-1/ }));
    fireEvent.change(screen.getByLabelText("Library view"), { target: { value: "missing" } });
    expect(await screen.findByText("Missing file · history retained")).toBeTruthy();
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("current-1")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /current-1.*sid-1/ })).toBeNull();
  });
  it("does not let an older current-files response replace a newer missing-history view", async () => {
    refreshed = true; render(<App />); await screen.findByRole("button", { name: /current-1.*sid-1/ });
    let finish; pendingCurrent = new Promise((resolve) => { finish = resolve; });
    fireEvent.click(screen.getByRole("button", { name: "Run search" }));
    fireEvent.change(screen.getByLabelText("Library view"), { target: { value: "missing" } });
    await screen.findByText("Missing file · history retained");
    await act(async () => { finish(response({ results: [item(4, "current")], total: 1, catalogue_status: "refreshed" })); });
    await waitFor(() => expect(screen.queryByRole("button", { name: /current-4/ })).toBeNull());
    expect(screen.getByText("Missing file · history retained")).toBeTruthy();
  });
  it("reports refresh failure in place without losing the stack or advising tensor installation", async () => {
    refreshed = true; failRefresh = true; render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /current-1.*sid-1/ }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh library" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "The local folder is unavailable; no changes were applied.");
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("current-1")).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/install.*torch/i);
  });
});
