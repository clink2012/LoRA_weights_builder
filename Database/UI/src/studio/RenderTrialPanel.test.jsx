import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import RenderTrialPanel from "./RenderTrialPanel";

const props = { apiBase: "/api", currentRecipe: { version_id: "recipe", name: "Saved pair", entries: [{ profile_version_id: "person-v1", stable_id: "person" }] }, selectedIds: ["person"], versionIds: { person: "person-v1" }, dirty: false, loading: false };
const generation = { checkpoint: "local-model", checkpoint_sha256: null, positive_prompt: "Dark dotted tights", negative_prompt: "", seed: "18446744073709551615", width: 896, height: 1152, steps: 20, sampler: "dpmpp_2m", scheduler: "sgm_uniform", guidance: 3.5, denoise: 1, stage: "first_pass" };
const record = { trial_id: "trial", name: "Original trial", created_at: "2026-10-06T12:00:00Z", receipt: { composition: { name: "Saved pair", historical_snapshot: { node_payloads: [{ loader_export: { numeric_csv: "1,0.25" } }] } }, criteria: "Keep the face and recover tights", declared_generation: generation }, evidence: [], assessments: [] };
const reply = (data) => Promise.resolve({ ok: true, json: async () => data });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
async function selectTrial() { await screen.findByText(/Original trial ·/); fireEvent.change(screen.getByLabelText("Saved trial"), { target: { value: "trial" } }); await screen.findByText("Original trial", { selector: "h3" }); }

describe("Render trial history", () => {
  it("pins only the recipe identifier and preserves a 64-bit seed as text", async () => {
    const fetch = vi.fn().mockImplementation((url) => url.includes("?limit") ? reply({ trials: [record] }) : reply(record));
    vi.stubGlobal("fetch", fetch);
    render(<RenderTrialPanel {...props} />);
    await selectTrial();
    fireEvent.click(screen.getByText("Use settings as comparison baseline"));
    await waitFor(() => expect(screen.getByLabelText("Fixed seed").value).toBe(generation.seed));
    fireEvent.change(screen.getByLabelText("Trial name"), { target: { value: "New variant" } });
    fireEvent.click(screen.getByText("Save trial record"));
    await waitFor(() => expect(fetch.mock.calls.some(([, options]) => options?.method === "POST")).toBe(true));
    const body = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === "POST")[1].body);
    expect(body.composition_version_id).toBe("recipe");
    expect(body.baseline_trial_id).toBe("trial");
    expect(body.generation.seed).toBe("18446744073709551615");
    expect(body.receipt).toBeUndefined();
  });

  it("blocks recording drafts or values different from the saved recipe", () => {
    vi.stubGlobal("fetch", vi.fn(() => reply({ trials: [] })));
    const { rerender } = render(<RenderTrialPanel {...props} dirty />);
    expect(screen.getByText("Save trial record").disabled).toBe(true);
    rerender(<RenderTrialPanel {...props} versionIds={{ person: "new-version" }} />);
    expect(screen.getByText("Save trial record").disabled).toBe(true);
    expect(screen.getByText(/Save the current selection and profile values/)).toBeTruthy();
  });

  it("requires evidence and saves no improvement without inventing identity acceptance", async () => {
    const attached = { ...record, evidence: [{ evidence_id: "image", filename: "result.png", sha256: "a".repeat(64) }] };
    const fetch = vi.fn().mockImplementation((url, options) => {
      if (url.includes("?limit")) return reply({ trials: [record] });
      if (options?.method === "POST") return reply(attached);
      return reply(record);
    });
    vi.stubGlobal("fetch", fetch);
    render(<RenderTrialPanel {...props} />); await selectTrial();
    expect(screen.getByText("Save assessment").disabled).toBe(true);
    const file = new File(["png"], "result.png", { type: "image/png" });
    fireEvent.change(screen.getByLabelText("Rendered PNG"), { target: { files: [file] } });
    fireEvent.click(screen.getByText("Attach PNG"));
    await screen.findByText("result.png");
    expect(screen.getByText("Recipe match unverified")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Intended effect"), { target: { value: "absent" } });
    fireEvent.change(screen.getByLabelText("Overall result"), { target: { value: "no_improvement" } });
    fireEvent.click(screen.getByText("Save assessment"));
    await waitFor(() => expect(fetch.mock.calls.some(([url]) => url.endsWith("/assessments"))).toBe(true));
    const body = JSON.parse(fetch.mock.calls.find(([url]) => url.endsWith("/assessments"))[1].body);
    expect(body.assessment.identity).toBe("not_assessed");
    expect(body.assessment.effect).toBe("absent");
    expect(body.assessment.outcome).toBe("no_improvement");
    expect(body.expected_assessment_id).toBe(null);
  });

  it("retains the idempotency key after an interrupted trial save", async () => {
    let attempts = 0;
    const fetch = vi.fn((url, options) => {
      if (url.includes("?limit")) return reply({ trials: [record] });
      if (!options) return reply(record);
      if (attempts++ === 0) return Promise.reject(new Error("network"));
      return reply(record);
    });
    vi.stubGlobal("fetch", fetch);
    render(<RenderTrialPanel {...props} />); await selectTrial();
    fireEvent.click(screen.getByText("Use settings as comparison baseline"));
    await waitFor(() => expect(screen.getByLabelText("Fixed seed").value).toBe(generation.seed));
    fireEvent.change(screen.getByLabelText("Trial name"), { target: { value: "Retry trial" } });
    fireEvent.click(screen.getByText("Save trial record"));
    await screen.findByText(/Connection interrupted/);
    fireEvent.click(screen.getByText("Save trial record"));
    await waitFor(() => expect(attempts).toBe(2));
    const bodies = fetch.mock.calls.filter(([, options]) => options?.method === "POST").map(([, options]) => JSON.parse(options.body));
    expect(bodies[0].idempotency_key).toBe(bodies[1].idempotency_key);
  });
});
