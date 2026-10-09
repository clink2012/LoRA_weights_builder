import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
const item = (id, status = "eligible") => ({ stable_id: id, filename: id, base_model_code: "FLX", presence: "current", compatibility: { status, reason: status === "eligible" ? "Pinned target matched" : "Unsupported tensor layout" } });
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
function setup(handler) {
  vi.stubGlobal("localStorage", { getItem: () => null, setItem: () => {} });
  const fetch = vi.fn(async (url, init) => {
    const path = new URL(url, "http://localhost").pathname;
    if (path.endsWith("/model-families")) return response({ families: [{ code: "FLX", display_name: "FLUX.1" }] });
    if (path.endsWith("/library-scan")) return response({ status: "idle" });
    if (path.endsWith("/composition-versions")) return response({ versions: [] });
    if (path.endsWith("/catalogue")) return response({ results: [item("Portrait"), item("Clothing")], total: 2, catalogue_status: "refreshed" });
    if (path.endsWith("/catalogue/compatible")) return handler(JSON.parse(init.body));
    return response({}, 404);
  }); vi.stubGlobal("fetch", fetch); return fetch;
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("structural candidate filtering", () => {
  it("uses server eligibility and filtered totals, reveals excluded reasons, and restores ordinary browsing after clearing", async () => {
    const fetch = setup((body) => response({ results: body.offset ? [item("Next candidate")] : body.view === "all" ? [item("Portrait"), item("Unknown", "unknown")] : [item("Portrait")], total: 51, counts: { eligible: 51, excluded: 12, unknown: 1 }, catalogue_status: "refreshed" }));
    render(<App />); fireEvent.click(await screen.findByRole("button", { name: /Portrait.*Portrait/ }));
    await screen.findByText("51 eligible · 12 excluded · 1 unverified");
    expect(screen.queryByRole("button", { name: /Clothing.*Clothing/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Next library page" })); await screen.findByRole("button", { name: /Next candidate.*Next candidate/ });
    expect(JSON.parse(fetch.mock.calls.filter(([url]) => url.endsWith("/compatible")).at(-1)[1].body).offset).toBe(50);
    fireEvent.click(screen.getByLabelText("Show excluded and unverified files"));
    await screen.findByText("Not established: Unsupported tensor layout");
    const unknown = screen.getByRole("button", { name: /Unknown.*Unknown/ });
    expect(unknown.disabled).toBe(true);
    fireEvent.click(unknown);
    expect(within(screen.getByRole("region", { name: "Selected stack" })).queryByText("Unknown")).toBeNull();
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("Portrait")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Clear stack" }));
    await screen.findByRole("button", { name: /Clothing.*Clothing/ });
    expect(screen.queryByLabelText("Show excluded and unverified files")).toBeNull();
  });
  it("keeps the stack but clears candidate claims when preflight fails", async () => {
    setup(() => response({ detail: { reason_code: "reference_not_supported", reason: "Reference layout has not been established." } }, 409));
    render(<App />); fireEvent.click(await screen.findByRole("button", { name: /Portrait.*Portrait/ }));
    await screen.findByText("Reference layout has not been established.");
    expect(screen.queryByRole("button", { name: /Clothing.*Clothing/ })).toBeNull();
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("Portrait")).toBeTruthy();
  });
  it("ignores an old reference response after the stack is cleared", async () => {
    let finish; setup(() => new Promise((resolve) => { finish = resolve; }));
    render(<App />); fireEvent.click(await screen.findByRole("button", { name: /Portrait.*Portrait/ }));
    await waitFor(() => expect(finish).toBeTypeOf("function"));
    fireEvent.click(screen.getByRole("button", { name: "Clear stack" })); await screen.findByRole("button", { name: /Clothing.*Clothing/ });
    await act(async () => finish(response({ results: [item("Old reference")], total: 1 })));
    expect(screen.queryByRole("button", { name: /Old reference/ })).toBeNull();
  });
  it("hides old candidates immediately during a delayed reference check and keeps failures closed", async () => {
    let finish; setup(() => new Promise((resolve) => { finish = resolve; }));
    render(<App />);
    const staleButton = await screen.findByRole("button", { name: /Clothing.*Clothing/ });
    fireEvent.click(screen.getByRole("button", { name: /Portrait.*Portrait/ }));
    expect(screen.queryByRole("button", { name: /Clothing.*Clothing/ })).toBeNull();
    fireEvent.click(staleButton);
    expect(within(screen.getByRole("region", { name: "Selected stack" })).queryByText("Clothing")).toBeNull();
    await act(async () => finish(response({ detail: "Cannot check this source" }, 503)));
    await screen.findByText("Cannot check this source");
    expect(screen.queryByRole("button", { name: /Clothing.*Clothing/ })).toBeNull();
  });
});
