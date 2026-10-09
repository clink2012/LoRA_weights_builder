import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CompositionPreferencePanel from "./CompositionPreferencePanel";

const target = "flux1-dev-native-v1";
const entries = [{ stable_id: "A", profile_version_id: "personal-A" }, { stable_id: "B", profile_version_id: "personal-B" }];
const recipe = { version_id: "recipe-one", name: "My preferred pair", target_contract_id: target, entries, historical_snapshot: { node_payloads: [{ loader_export: { numeric_csv: "never copy" } }] }, requires_revalidation: true };
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const props = () => ({ apiBase: "/api", selectedIds: ["A", "B"], versionIds: { A: "default-A", B: "default-B" }, currentRecipe: null, result: null, dirty: false, loading: false, onBusyChange: vi.fn(), onRestore: vi.fn(), onInvalidatePrepared: vi.fn() });
const preferred = { status: "preferred", stable_ids: ["A", "B"], target_contract_id: target, recipe };

describe("Preferred composition recall", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(preferred))));
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("loads the exact preferred recipe only after an explicit decision", async () => {
    const p = props(); const { rerender } = render(<CompositionPreferencePanel {...p} />);
    expect(fetch).not.toHaveBeenCalled(); expect(p.onRestore).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Load preferred recipe' }));
    await waitFor(() => expect(p.onRestore).toHaveBeenCalledWith({ ...recipe, recalled_preference: true }));
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ stable_ids: ["A", "B"], target_contract_id: target });
    rerender(<CompositionPreferencePanel {...p} versionIds={{ A: "other", B: "default-B" }} />);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Copy full vector" })).toBeNull();
  });

  it.each(["dirty", "version", "recipe", "selection"])("ignores a late manual load after %s changes", async (mode) => {
    let finish; fetch.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const p = props(); const { rerender } = render(<CompositionPreferencePanel {...p} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load preferred recipe' }));
    const changed = mode === "dirty" ? { dirty: true } : mode === "version" ? { versionIds: { A: "manual-A", B: "default-B" } } : mode === "recipe" ? { currentRecipe: recipe } : mode === "selection" ? { selectedIds: ["B", "A"] } : { loading: true };
    const oldFinish = finish;
    rerender(<CompositionPreferencePanel {...p} {...changed} />);
    if (mode === "loading-round-trip") rerender(<CompositionPreferencePanel {...p} loading={false} />);
    await act(async () => oldFinish(response(preferred)));
    expect(p.onRestore).not.toHaveBeenCalled();
  });

  it("shows changed-source review information without applying the retained recipe", async () => {
    fetch.mockResolvedValue(response({ ...preferred, status: "needs_review", recipe: null, reason: "Source changed; history retained." }));
    const p = props(); render(<CompositionPreferencePanel {...p} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load preferred recipe' }));
    await screen.findByText("Source changed; history retained.");
    expect(p.onRestore).not.toHaveBeenCalled();
  });

  it("rejects a mismatched loader order", async () => {
    fetch.mockResolvedValue(response({ ...preferred, stable_ids: ["B", "A"] }));
    const p = props(); render(<CompositionPreferencePanel {...p} />);
    fireEvent.click(screen.getByRole('button', { name: 'Load preferred recipe' }));
    await screen.findByText(/does not match this ordered combination/);
    expect(p.onRestore).not.toHaveBeenCalled();
  });

  it("makes only the prepared matching recipe preferred and preserves the current stack", async () => {
    const p = { ...props(), currentRecipe: recipe, versionIds: { A: "personal-A", B: "personal-B" }, result: { preparation_digest: "current-digest", compatible: true } };
    render(<CompositionPreferencePanel {...p} />);
    expect(fetch).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Make this recipe preferred" }));
    await screen.findByText(/This recipe is preferred/);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ version_id: "recipe-one", expected_preparation_digest: "current-digest" });
    expect(p.onRestore).not.toHaveBeenCalled();
    expect(p.onBusyChange.mock.calls).toEqual([[true], [false]]);
  });

  it("invalidates copying on a preference conflict", async () => {
    fetch.mockResolvedValue(response({ detail: "Preparation changed" }, 409));
    const p = { ...props(), currentRecipe: recipe, versionIds: { A: "personal-A", B: "personal-B" }, result: { preparation_digest: "old" } };
    render(<CompositionPreferencePanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Make this recipe preferred" }));
    await screen.findByRole("alert");
    expect(p.onInvalidatePrepared).toHaveBeenCalledTimes(1);
  });

  it("returns to originals without sending personal values and invalidates copying immediately", async () => {
    const original = { status: "originals", stable_ids: ["A", "B"], target_contract_id: target, entries: [{ stable_id: "A", profile_version_id: "default-A" }, { stable_id: "B", profile_version_id: "default-B" }], requires_revalidation: true };
    fetch.mockResolvedValue(response(original));
    const p = { ...props(), currentRecipe: recipe }; render(<CompositionPreferencePanel {...p} />);
    fireEvent.click(screen.getByRole("button", { name: "Return combination to original values" }));
    expect(p.onInvalidatePrepared).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(p.onRestore).toHaveBeenCalledWith(original));
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ stable_ids: ["A", "B"], target_contract_id: target });
  });

  it("gates preference actions while drafts exist", () => {
    const p = { ...props(), currentRecipe: recipe, dirty: true };
    render(<CompositionPreferencePanel {...p} />);
    expect(screen.getByRole("button", { name: "Return combination to original values" }).disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Make this recipe preferred" }).disabled).toBe(true);
    expect(fetch).not.toHaveBeenCalled();
  });
});
