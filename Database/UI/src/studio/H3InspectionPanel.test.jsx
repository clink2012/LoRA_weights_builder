import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import H3InspectionPanel from "./H3InspectionPanel";
import Studio from "./Studio";

const item = { stable_id: "MH3-PPL-1", filename: "H3 person.safetensors", base_model_code: "MH3" };
const observed = { status: "observed", pair_count: 3, tensor_count: 6, accounted_tensor_count: 6, issue_count: 0, issues: [],
  slots: [{ group: "main", index: 0, label: "MAIN 0", pair_count: 2, ranks: [4] }, { group: "refiner", index: 1, label: "REFINER 1", pair_count: 1, ranks: [2] }],
  file_identity: { header_sha256: "test-header-fingerprint", tensor_payload_read: false } };
const response = (data, status = 200) => ({ ok: status < 400, json: async () => data });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows coverage counts and ranks without presenting editable weights or export", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => response(observed)));
  const close = vi.fn(); render(<H3InspectionPanel apiBase="/api" item={item} onClose={close} />);
  expect(await screen.findByRole("img", { name: "H3 observed pair counts by block" })).toBeTruthy();
  expect(screen.getByRole("status").textContent).toContain("3 ordinary pairs");
  expect(screen.getByText("test-header-fingerprint")).toBeTruthy();
  expect(screen.queryAllByRole("spinbutton")).toHaveLength(0);
  expect(screen.queryByRole("button", { name: /copy/i })).toBeNull();
  expect(screen.getByText(/These are not block weights or measured contributions/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Close inspection" })); expect(close).toHaveBeenCalledOnce();
});

it.each([
  [response({ status: "unavailable", reason: "Adapter is missing", slots: [] }), "Adapter is missing"],
  [response({ detail: "Unknown catalogue ID" }, 404), "Unknown catalogue ID"],
])("explains unavailable evidence", async (reply, text) => {
  vi.stubGlobal("fetch", vi.fn(async () => reply));
  render(<H3InspectionPanel apiBase="/api" item={item} onClose={() => {}} />);
  expect((await screen.findByRole("alert")).textContent).toBe(text);
  expect(screen.queryByRole("img")).toBeNull();
});

it("aborts the prior request and excludes a late response after changing adapters", async () => {
  let finish; const fetcher = vi.fn(() => new Promise((resolve) => { finish = resolve; }));
  vi.stubGlobal("fetch", fetcher);
  const { rerender } = render(<H3InspectionPanel key="first" apiBase="/api" item={item} onClose={() => {}} />);
  const oldFinish = finish;
  fetcher.mockImplementation(async () => response({ ...observed, pair_count: 8 }));
  rerender(<H3InspectionPanel key="second" apiBase="/api" item={{ ...item, stable_id: "MH3-PPL-2" }} onClose={() => {}} />);
  await screen.findByText(/8 ordinary pairs/);
  await act(async () => oldFinish(response(observed)));
  expect(screen.queryByText(/3 ordinary pairs/)).toBeNull();
  expect(fetcher.mock.calls[0][1].signal.aborted).toBe(true);
});

function studioProps(selectedItems = []) {
  return { apiBase: "/api", libraryFamily: "MH3", catalog: [item], selectedItems, selectedIds: selectedItems.map((entry) => entry.stable_id), computedById: new Map(), versionIds: {}, draftProfiles: {}, result: null, loading: false, libraryPresence: "current", catalogueStatus: "refreshed", scan: { status: { status: "idle" } }, search: "", page: 0, pages: 1, onToggle: vi.fn(), onCalculate: vi.fn() };
}

it.each([false, true])("inspects H3 independently of an existing FLUX stack (%s)", async (withFlux) => {
  vi.stubGlobal("fetch", vi.fn(async (url) => String(url).includes("h3-inspection") ? response(observed) : response({ versions: [], trials: [], recipes: [], total: 0 })));
  const flux = { stable_id: "FLX-PPL-1", filename: "Flux person.safetensors", base_model_code: "FLX" };
  const props = studioProps(withFlux ? [flux] : []);
  render(<Studio {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "Inspect H3 H3 person" }));
  expect(await screen.findByRole("img", { name: "H3 observed pair counts by block" })).toBeTruthy();
  expect(props.onToggle).not.toHaveBeenCalled(); expect(props.onCalculate).not.toHaveBeenCalled();
  const stack = screen.getByRole("region", { name: "Selected stack" });
  expect(stack.textContent).toContain(withFlux ? "FLUX.1 dev · Inspire block loader" : "MiniMax H3 · adapter inspection");
  if (withFlux) expect(stack.textContent).toContain("Flux person");
  else expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(true);
  expect(globalThis.fetch.mock.calls.every(([, init]) => !init?.method || init.method === "GET")).toBe(true);
});
