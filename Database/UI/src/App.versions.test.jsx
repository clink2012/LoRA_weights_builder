import { render, screen, fireEvent, waitFor, cleanup, within } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import App from "./App";

const labels = ["BASE", ...Array.from({ length: 19 }, (_, n) => `DOUBLE ${n}`), ...Array.from({ length: 38 }, (_, n) => `SINGLE ${n}`)];
const root = () => ({ version_id: "default-1", default_id: "default-1", parent_id: null, kind: "default", name: "Default", sequence: 1, stable_id: "sid-1", created_at: "2026-10-03T20:00:00Z", binding: { slots: labels.map((label) => ({ label, group: label.split(" ")[0] })) }, values: labels.map(() => 1), settings: { role: "person", strength_model: 1, strength_clip: null, affect_clip: false }, ab: {} });
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });

// These full workflows render 58 slots across several API round trips. Shared
// CI CPUs need more headroom than isolated unit tests; assertions stay unchanged.
describe("Versioned Studio workflow", { timeout: 15_000 }, () => {
  let versions, selectedId, recipes, failSave, lastPreparation, deferLoad, deferSave, finishPending, recipeConflict, preferredRecipe;
  beforeEach(() => {
    versions = [root()]; selectedId = "default-1"; recipes = []; failSave = false; deferLoad = false; deferSave = false; recipeConflict = false;
    preferredRecipe = null;
    const stored = new Map(); vi.stubGlobal("localStorage", { getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value) });
    vi.stubGlobal("fetch", vi.fn(async (input, init) => {
      const url = new URL(String(input), "http://localhost"); const path = url.pathname; const body = init?.body ? JSON.parse(init.body) : null;
      if (path.endsWith("model-families")) return response({ families: [{ code: "FLX", display_name: "FLUX.1", support_level: "experimental" }] });
      if (path.endsWith("lora/search") || path.endsWith("/catalogue") || path.endsWith("/catalogue/compatible")) return response({ results: [{ id: 1, stable_id: "sid-1", filename: "Portrait.safetensors", base_model_code: "FLX", block_layout: "flux_transformer_57", role: "person" }], total: 1 });
      if (path.endsWith("/library-scan")) return response(body ? { status: "complete", phase: "finished", catalogue_scan_id: "scan-1", catalogue: { counts: { present: 1, added: 0, missing: 0 } } } : { status: "idle" });
      if (path.endsWith("/composition-preferences/resolve")) return response({ status: preferredRecipe ? "preferred" : "none", stable_ids: body.stable_ids, target_contract_id: body.target_contract_id, recipe: preferredRecipe });
      if (path.endsWith("/composition-preferences/choose")) { preferredRecipe = recipes.find((entry) => entry.version_id === body.version_id); return response({ status: "preferred", stable_ids: preferredRecipe.entries.map((entry) => entry.stable_id), target_contract_id: preferredRecipe.target_contract_id, recipe: preferredRecipe }); }
      if (path.endsWith("/composition-preferences/originals")) { preferredRecipe = null; return response({ status: "originals", ...body, entries: body.stable_ids.map((id) => ({ stable_id: id, profile_version_id: "default-1" })), requires_revalidation: true }); }
      if (path.endsWith("/defaults")) return response(versions[0]);
      if (path.endsWith("/selection")) { if (body) selectedId = body.version_id; return response(versions.find((version) => version.version_id === selectedId)); }
      if (path.endsWith("/revisions")) {
        if (failSave) return response({ detail: "Database is temporarily unavailable" }, 503);
        const saved = { ...versions[0], ...body, version_id: `personal-${versions.length}`, kind: "personal", sequence: versions.length + 1 };
        versions.push(saved); if (deferSave) return new Promise((resolve) => { finishPending = () => resolve(response(saved)); }); return response(saved);
      }
      if (path.includes("/profile-versions/sid-1/versions/")) return response(versions.find((version) => version.version_id === path.split("/").at(-1)));
      if (path.endsWith("/profile-versions/sid-1")) return response({ versions });
      if (path.endsWith("/prepare-blocks")) {
        const version = versions.find((entry) => entry.version_id === body.profile_version_ids?.["sid-1"]) || versions[0];
        lastPreparation = { compatible: true, preparation_digest: `digest-${version.version_id}`, node_payloads: [{ stable_id: "sid-1", filename: "Portrait.safetensors", profile_version_id: version.version_id, profile_name: version.name, strength_model: version.settings.strength_model, strength_clip: null, loader_export: { status: "ready", adapter_id: "inspire_flux1_v1", recommendation_basis: version.kind === "default" ? "structural_baseline_unvalidated" : "manual_variant_unvalidated", numeric_csv: version.values.join(","), slot_values: version.values, slot_labels: labels, architecture_slot_values: version.values, architecture_slot_labels: labels, loader_slot_count: 58, architecture_slot_count: 58 } }] };
        return response(lastPreparation);
      }
      if (path.includes("/analysis-jobs")) return response({ job_id: "job-tabs", status: "running", entries: [{ stable_id: "sid-1", profile_version_id: selectedId }], target_contract_id: "flux1-dev-native-v1", preparation_digest: `digest-${selectedId}`, metrics: null });
      if (path.endsWith("/composition-versions")) {
        if (!body) return response({ versions: recipes });
        if (recipeConflict) return response({ detail: "Preparation changed. Prepare again." }, 409);
        const recipe = { ...body, version_id: `recipe-${recipes.length}`, created_at: "2026-10-03T20:00:00Z", historical_snapshot: lastPreparation, requires_revalidation: true }; recipes.push(recipe); return response(recipe);
      }
      if (path.includes("/composition-versions/")) { const recipe = recipes.find((recipe) => recipe.version_id === path.split("/").at(-1)); if (deferLoad) return new Promise((resolve) => { finishPending = () => resolve(response(recipe)); }); return response(recipe); }
      return response({}, 404);
    }));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
  async function start() {
    render(<App />); fireEvent.click(await screen.findByRole("button", { name: /Portrait.*sid-1/ }));
    fireEvent.click(screen.getByRole("button", { name: "Prepare block values" })); await screen.findByRole("button", { name: "Copy full vector" });
    fireEvent.click(screen.getByRole("button", { name: "Open variants & history" })); await screen.findByRole("region", { name: "Variants and history" });
    fireEvent.click(screen.getByText("Saved compositions"));
  }
  function edit(value = "0.35") {
    fireEvent.change(screen.getByLabelText("New exact value"), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: "Apply block value" }));
  }
  async function save(name = "Soft portrait") {
    fireEvent.change(screen.getByLabelText("Revision name"), { target: { value: name } });
    fireEvent.click(screen.getByRole("button", { name: "Save new revision" }));
    await waitFor(() => expect(screen.queryByText("Personal draft")).toBeNull());
  }
  async function prepare() { await waitFor(() => expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(false)); fireEvent.click(screen.getByRole("button", { name: "Prepare block values" })); await screen.findByRole("button", { name: "Copy full vector" }); }

  it("saves exact BASE edits as a new revision, invalidates copying, and restores Default without removing history", async () => {
    await start(); await prepare(); edit("-0.2345");
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(true);
    await save(); expect(versions).toHaveLength(2); expect(versions[0].values[0]).toBe(1); expect(versions[1].values[0]).toBe(-0.2345);
    await prepare(); expect(screen.getByRole("textbox", { name: "Full block values for Portrait" }).value.split(",")[0]).toBe("-0.2345");
    fireEvent.click(screen.getByRole("button", { name: /Default.*Version 1/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /Default.*Version 1/ }).getAttribute("aria-pressed")).toBe("true"));
    expect(screen.getByRole("button", { name: /Soft portrait.*Version 2/ })).toBeTruthy();
    await prepare(); expect(screen.getByRole("textbox", { name: "Full block values for Portrait" }).value.split(",")[0]).toBe("1");
  });
  it("preserves a failed-save draft and saves explicit personal A/B bounds", async () => {
    await start(); edit("0.35");
    fireEvent.click(screen.getByText("Personal A/B experiment"));
    fireEvent.change(screen.getByLabelText("Minimum"), { target: { value: "0.2" } });
    fireEvent.change(screen.getByLabelText("Maximum"), { target: { value: "0.5" } });
    fireEvent.click(screen.getByRole("button", { name: "Set trial range" }));
    failSave = true; fireEvent.change(screen.getByLabelText("Revision name"), { target: { value: "Trial A" } }); fireEvent.click(screen.getByRole("button", { name: "Save new revision" }));
    expect(await screen.findByText("Database is temporarily unavailable")).toBeTruthy();
    expect(screen.getByLabelText("Revision name").value).toBe("Trial A"); expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    failSave = false; await save("Trial A");
    expect(versions[1].ab.A).toMatchObject({ slot_labels: ["BASE"], value: 0.35, min: 0.2, max: 0.5 });
  });
  it("preserves the selected stack and exact personal draft through catalogue filtering and refresh", async () => {
    await start(); edit("-0.4567");
    fireEvent.change(screen.getByLabelText("Revision name"), { target: { value: "Keep this draft" } });
    fireEvent.change(screen.getByLabelText("Library view"), { target: { value: "all" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Search", exact: true }).disabled).toBe(false));
    expect(screen.getByLabelText("Revision name").value).toBe("Keep this draft");
    fireEvent.click(screen.getByRole("button", { name: "Refresh library" }));
    await screen.findByText(/Library refreshed: 1 current/);
    expect(within(screen.getByRole("region", { name: "Selected stack" })).getByText("Portrait")).toBeTruthy();
    expect(screen.getByLabelText("Revision name").value).toBe("Keep this draft");
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    await save("Keep this draft");
    expect(versions[1].values[0]).toBe(-0.4567);
    expect(globalThis.fetch.mock.calls.filter(([url, init]) => String(url).endsWith("/library-scan") && init?.method === "POST")).toHaveLength(1);
  });
  it("keeps exact drafts and recipe fields mounted while switching workspace tabs", async () => {
    await start(); edit("-0.456789");
    fireEvent.change(screen.getByLabelText("Revision name"), { target: { value: "Keep across tabs" } });
    fireEvent.change(screen.getByLabelText("Recipe name"), { target: { value: "Future recipe" } });
    const editor = screen.getByLabelText("New exact value");
    const compare = screen.getByRole("tab", { name: "Compare & experiment" });
    fireEvent.click(compare);
    expect(document.getElementById("studio-panel-build").hidden).toBe(true);
    expect(editor.isConnected).toBe(true);
    expect(screen.getByRole("button", { name: "Measure current sources" }).disabled).toBe(true);
    fireEvent.keyDown(compare, { key: "Home" });
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Build", exact: true }));
    expect(screen.getByLabelText("New exact value")).toBe(editor);
    expect(screen.getByLabelText("Revision name").value).toBe("Keep across tabs");
    expect(screen.getByLabelText("Recipe name").value).toBe("Future recipe");
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    await save("Keep across tabs"); expect(versions[1].values[0]).toBe(-0.456789);
  });
  it("keeps a running measurement mounted when returning to Build", async () => {
    await start(); await prepare();
    fireEvent.click(screen.getByRole("tab", { name: "Compare & experiment" }));
    fireEvent.click(screen.getByRole("button", { name: "Measure current sources" }));
    await screen.findByText("Reading tensors and measuring parameter updates…");
    const panel = document.querySelector(".studio-measurements");
    fireEvent.click(screen.getByRole("tab", { name: "Build", exact: true }));
    expect(panel.isConnected).toBe(true);
    expect(screen.getByRole("button", { name: "Copy full vector" })).toBeTruthy();
    expect(globalThis.fetch.mock.calls.some(([url]) => String(url).endsWith("/job-tabs/cancel"))).toBe(false);
    fireEvent.click(screen.getByRole("tab", { name: "Compare & experiment" }));
    expect(screen.getByRole("button", { name: "Cancel measurement" })).toBeTruthy();
    expect(document.querySelector(".studio-measurements")).toBe(panel);
  });
  it("saves role corrections and exact supporting model strength in a personal revision", async () => {
    await start(); fireEvent.click(screen.getByText("LoRA role and supporting settings"));
    fireEvent.change(screen.getByLabelText("LoRA role"), { target: { value: "clothing" } });
    fireEvent.change(screen.getByLabelText("Model strength"), { target: { value: "0.712345" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply model strength" }));
    await save("Clothing role");
    expect(versions[1].settings).toMatchObject({ role: "clothing", strength_model: 0.712345 });
    await prepare(); expect(screen.getByText(/Model strength 0.712345/)).toBeTruthy();
  });
  it("gates edits and stack mutations while a saved recipe is loading", async () => {
    recipes = [{ version_id: "recipe-0", name: "Earlier recipe", created_at: "2026-10-03T20:00:00Z", entries: [{ stable_id: "sid-1", profile_version_id: "default-1" }], historical_snapshot: {} }];
    await start(); deferLoad = true;
    fireEvent.change(screen.getByLabelText("Saved recipe"), { target: { value: "recipe-0" } }); fireEvent.click(screen.getByRole("button", { name: "Load recipe" }));
    expect(screen.getByLabelText("New exact value").disabled).toBe(true);
    expect(screen.getByRole("button", { name: /Portrait.*sid-1/ }).disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Clear stack" }).disabled).toBe(true);
    finishPending(); await waitFor(() => expect(screen.queryByRole("region", { name: "Variants and history" })).toBeNull());
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
  });
  it("prevents A/B removal or name changes during a pending save", async () => {
    await start(); edit(); fireEvent.click(screen.getByText("Personal A/B experiment"));
    fireEvent.click(screen.getByRole("button", { name: "Set trial range" }));
    fireEvent.change(screen.getByLabelText("Revision name"), { target: { value: "Pending trial" } });
    deferSave = true; fireEvent.click(screen.getByRole("button", { name: "Save new revision" }));
    expect(screen.getByRole("button", { name: "Remove A" }).disabled).toBe(true);
    expect(screen.getByLabelText("Revision name").disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Prepare block values" }).disabled).toBe(true);
    finishPending(); await waitFor(() => expect(screen.queryByText("Personal draft")).toBeNull());
    expect(versions[1].ab.A.value).toBe(0.35);
  });
  it("removes copy authority when recipe save detects changed preparation", async () => {
    await start(); await prepare(); recipeConflict = true;
    fireEvent.change(screen.getByLabelText("Recipe name"), { target: { value: "Changed source" } });
    fireEvent.click(screen.getByRole("button", { name: "Save recipe version" }));
    await screen.findByText("Preparation changed. Prepare again.");
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
  });
  it("loads the pinned recipe version and never turns a historical snapshot into a fresh export", async () => {
    await start(); edit("0.35"); await save(); await prepare();
    fireEvent.change(screen.getByLabelText("Recipe name"), { target: { value: "Portrait recipe" } });
    fireEvent.click(screen.getByRole("button", { name: "Save recipe version" })); await screen.findByText("Recipe saved as a new immutable version. Previous recipes are preserved.");
    expect(recipes[0].entries).toEqual([{ stable_id: "sid-1", profile_version_id: "personal-1" }]);
    fireEvent.click(screen.getByRole("button", { name: /Default.*Version 1/ })); await waitFor(() => expect(selectedId).toBe("default-1"));
    fireEvent.change(screen.getByLabelText("Saved recipe"), { target: { value: "recipe-0" } }); fireEvent.click(screen.getByRole("button", { name: "Load recipe" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Variants and history" })).toBeNull());
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open variants & history" }));
    const panel = await screen.findByRole("region", { name: "Variants and history" });
    expect(within(panel).getByRole("heading", { name: "Soft portrait" })).toBeTruthy();
    await prepare(); expect(screen.getByRole("textbox", { name: "Full block values for Portrait" }).value.split(",")[0]).toBe("0.35");
  });
  it("recalls preferred personal values on reselection and returns to originals without deleting history", async () => {
    await start(); edit("-0.3456789"); await save("My preferred portrait"); await prepare();
    fireEvent.change(screen.getByLabelText("Recipe name"), { target: { value: "Personal portrait recipe" } });
    fireEvent.click(screen.getByRole("button", { name: "Save recipe version" }));
    await screen.findByText("Recipe saved as a new immutable version. Previous recipes are preserved.");
    fireEvent.click(screen.getByRole("button", { name: "Make this recipe preferred" }));
    await screen.findByText(/This recipe is preferred/);
    fireEvent.click(screen.getByRole("button", { name: "Clear stack" }));
    fireEvent.click(screen.getByRole("button", { name: /Portrait.*sid-1/ }));
    await screen.findByText(/Preferred composition recalled: Personal portrait recipe/);
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    await prepare();
    expect(screen.getByRole("textbox", { name: "Full block values for Portrait" }).value.split(",")[0]).toBe("-0.3456789");
    fireEvent.click(screen.getByRole("button", { name: "Return combination to original values" }));
    await waitFor(() => expect(preferredRecipe).toBeNull());
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
    await prepare();
    expect(screen.getByRole("textbox", { name: "Full block values for Portrait" }).value.split(",")[0]).toBe("1");
    expect(versions).toHaveLength(2); expect(recipes).toHaveLength(1);
  });
});
