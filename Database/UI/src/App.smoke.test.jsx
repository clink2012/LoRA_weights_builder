import { act, render, screen, fireEvent, waitFor, cleanup, within } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import App from "./App";

const reply = (data, ok = true) => ({ ok, status: ok ? 200 : 400, json: async () => data });
const item = (id) => ({ id, stable_id: `sid-${id}`, filename: `demo-${id}.safetensors`, base_model_code: "FLX", category_code: "STL", role: "style", compatibility: { status: "eligible" }, has_block_weights: true, block_layout: id === 2 ? "flux_fallback_16" : "flux_transformer_57" });
const labels = ["BASE", ...Array.from({ length: 19 }, (_, n) => `DOUBLE ${n}`), ...Array.from({ length: 38 }, (_, n) => `SINGLE ${n}`)];
const values = labels.map((_, n) => n === 1 ? -0.1234 : 0.5678);
const csv = values.map((value) => value.toFixed(4)).join(",");
function payload(id, ready = true) {
  return { stable_id: `sid-${id}`, strength_model: 0.8123, strength_clip: null, block_weights: [1, 1], role_strength_recommendation: { recommended_model_strength: 0.2 }, loader_export: { status: ready ? "ready" : "blocked", reason: "Exact adapter coverage is missing.", adapter_id: "inspire_flux1_v1", numeric_csv: ready ? csv : null, slot_values: ready ? values : [], slot_labels: ready ? labels : [], architecture_slot_count: 58, architecture_slot_values: values, architecture_slot_labels: labels, loader_slot_count: ready ? 58 : 0 } };
}

describe("Studio integration", () => {
  let mode;
  let resolveCombine;
  let resolveRefresh;
  beforeEach(() => {
    mode = "ready";
    const stored = new Map();
    vi.stubGlobal("localStorage", { getItem: (key) => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) });
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } });
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/library-scan") && !init?.method) return reply({ status: "idle" });
      if (url.endsWith("/library-scan")) return new Promise((resolve) => { resolveRefresh = () => resolve(reply({ status: "complete", phase: "finished", catalogue_scan_id: "scan-1", catalogue: { counts: { present: 2, added: 0, missing: 0 } } })); });
      if (url.endsWith("/composition-versions")) return reply({ versions: [] });
      if (url.endsWith("/model-families")) return reply({ families: [{ code: "FLX", display_name: "Flux", support_level: "mixed-scanned-fallback" }, { code: "MH3", display_name: "MiniMax H3", support_level: "metadata-only" }] });
      if ((url.includes("/lora/search") || url.includes("/catalogue?") || url.endsWith("/catalogue/compatible"))) {
        if (url.endsWith("/catalogue/compatible") && ["reference-failed", "budget-failed"].includes(mode)) return { ok: false, status: 409, json: async () => ({ detail: { reason_code: mode === "reference-failed" ? "reference_not_supported" : "budget_exceeded", reason: "Preflight needs attention" } }) };
        const page = init?.body ? String(JSON.parse(init.body).offset) : new URL(url, "http://localhost").searchParams.get("offset");
        return reply({ results: page === "50" ? [item(3)] : [item(1), item(2)], total: 51 });
      }
      if (url.endsWith("/lora/prepare-blocks") && init?.method === "POST") {
        const ids = JSON.parse(init.body).stable_ids;
        const result = { compatible: true, validated_base_model: "FLX", node_payloads: ids.map((id) => payload(Number(id.slice(4)), mode !== "blocked")), warnings: [] };
        if (mode === "deferred") return new Promise((resolve) => { resolveCombine = () => resolve(reply(result)); });
        if (mode === "error") return reply({ detail: { compatible: false, reasons: ["Wrong model family"], warnings: ["Incompatible stack"], node_payloads: [payload(1)] } }, false);
        return reply(result);
      }
      return reply({}, false);
    });
  });
  afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
  async function choose(id = 1) { const button = await screen.findByRole("button", { name: new RegExp(`demo-${id}.*sid-${id}`) }); await waitFor(() => expect(button.disabled).toBe(false)); fireEvent.click(button); }
  async function calculate() { fireEvent.click(screen.getByRole("button", { name: "Prepare block values" })); await screen.findByRole("region", { name: "Full loader vectors" }); }

  it("copies the complete backend vector without rounding or applying advisory strengths", async () => {
    render(<App />); await choose(); await calculate();
    expect(screen.getByRole("textbox", { name: "Full block values for demo-1" }).value).toBe(csv);
    expect(csv.split(",")).toHaveLength(58);
    expect(screen.getByText(/Model strength 0.8123/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "DOUBLE 0: -0.1234" }));
    expect(screen.getByRole("spinbutton", { name: "Exact value" }).value).toBe("-0.1234");
    fireEvent.click(screen.getByRole("button", { name: "Copy full vector" }));
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith(csv));
  });
  it("removes the prior export immediately while preparing again", async () => {
    render(<App />); await choose(); await calculate();
    expect(screen.getByRole("button", { name: "Copy full vector" })).toBeTruthy();
    mode = "deferred";
    fireEvent.click(screen.getByRole("button", { name: "Prepare block values" }));
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    resolveCombine();
    await screen.findByRole("button", { name: "Copy full vector" });
  });
  it("does not expose legacy or blocked CSV as copyable loader output", async () => {
    mode = "blocked"; render(<App />); await choose(); await calculate();
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    expect(screen.getAllByText("Exact adapter coverage is missing.").length).toBeGreaterThan(0);
  });
  it("keeps copy unavailable throughout refresh, including old and attempted new preparation", async () => {
    render(<App />); await choose(); await calculate();
    mode = "deferred";
    fireEvent.click(screen.getByRole("button", { name: "Prepare block values" }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh library" }));
    expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(true);
    await act(async () => { resolveCombine(); });
    await waitFor(() => expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull());
    const calls = globalThis.fetch.mock.calls.filter(([url]) => String(url).endsWith("/lora/prepare-blocks")).length;
    fireEvent.click(screen.getByRole("button", { name: "Prepare block values" }));
    expect(globalThis.fetch.mock.calls.filter(([url]) => String(url).endsWith("/lora/prepare-blocks"))).toHaveLength(calls);
    resolveRefresh();
    await waitFor(() => expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(false));
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    mode = "ready"; await calculate();
    expect(screen.getByRole("button", { name: "Copy full vector" })).toBeTruthy();
  });
  it("keeps server-eligible LoRAs visible despite different historical layouts", async () => {
    render(<App />); await choose(); await choose(2);
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("demo-2")).toBeTruthy();
  });
  it("preserves selection and chain order across catalogue pages", async () => {
    render(<App />); await choose();
    await waitFor(() => expect(screen.getByRole("button", { name: "Next library page" }).disabled).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Next library page" })); await choose(3);
    const stack = screen.getByRole("region", { name: "Selected stack" });
    expect(within(stack).getByText("demo-1")).toBeTruthy(); expect(within(stack).getByText("demo-3")).toBeTruthy();
    await calculate();
    const call = globalThis.fetch.mock.calls.find(([url]) => String(url).endsWith("/lora/prepare-blocks"));
    expect(JSON.parse(call[1].body).stable_ids).toEqual(["sid-1", "sid-3"]);
  });
  it("discards a response calculated before the selected stack changed", async () => {
    mode = "deferred"; render(<App />); await choose();
    fireEvent.click(screen.getByRole("button", { name: "Prepare block values" })); await choose(2);
    resolveCombine();
    await waitFor(() => expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(false));
    expect(screen.queryByRole("region", { name: "Full loader vectors" })).toBeNull();
  });
  it("keeps registry support labels and remembers the chosen theme", async () => {
    render(<App />);
    await waitFor(() => expect(Array.from(screen.getByLabelText("Base model").options).map((option) => option.textContent)).toContain("MiniMax H3 · metadata only"));
    fireEvent.click(screen.getByRole("button", { name: "Switch colour theme" }));
    expect(document.querySelector(".lm-app").dataset.theme).toBe("carbon");
    expect(localStorage.getItem("lora-studio-theme")).toBe("carbon");
  });
  it("migrates the earlier theme preference and keeps the filter panel collapsed after remount", async () => {
    localStorage.setItem("lora-studio-theme", "atelier");
    const view = render(<App />);
    expect(document.querySelector(".lm-app").dataset.theme).toBe("carbon");
    fireEvent.click(screen.getByRole("button", { name: "Hide library filters" }));
    expect(document.getElementById("library-filter-menu").hidden).toBe(true);
    expect(localStorage.getItem("lora-studio-filters-collapsed")).toBe("true");
    view.unmount(); render(<App />);
    expect(screen.getByRole("button", { name: "Show library filters" }).getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(screen.getByRole("button", { name: "Show library filters" }));
    expect(document.getElementById("library-filter-menu").hidden).toBe(false);
    await screen.findByRole("button", { name: /demo-1.*sid-1/ });
  });
  it("invalidates prepared copy on current reference failure but not an ordinary candidate-budget error", async () => {
    render(<App />); await choose(); await calculate();
    mode = "budget-failed"; fireEvent.click(screen.getByLabelText("Show excluded and unverified files"));
    await screen.findByText("Preflight needs attention");
    expect(screen.getByRole("button", { name: "Copy full vector" })).toBeTruthy();
    mode = "reference-failed"; fireEvent.click(screen.getByLabelText("Show excluded and unverified files"));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull());
  });
  it("shows structured errors and keeps copying unavailable", async () => {
    mode = "error"; render(<App />); await choose(); await calculate();
    expect(screen.getByRole("alert").textContent).toBe("Incompatible stack");
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
  });
});

