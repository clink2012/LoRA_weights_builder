import { render, screen, fireEvent, waitFor, within, cleanup } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import App from "./App";

function jsonResponse(data, ok = true, status = 200) {
  return {
    ok,
    status,
    async json() {
      return data;
    },
  };
}

describe("App combine smoke", () => {
  let combineMode = "success";

  beforeEach(() => {
    combineMode = "success";
    globalThis.fetch = vi.fn(async (input, init) => {
      const url = String(input);

      if (url.includes("/lora/search")) {
        return jsonResponse({
          results: [
            {
              id: 1,
              stable_id: "sid-1",
              filename: "demo-lora-1.safetensors",
              file_path: "/tmp/demo-lora-1.safetensors",
              base_model_code: "FLX",
              category_code: "STL",
              role: "style",
              has_block_weights: true,
              block_layout: "flux_fallback_16",
            },
            {
              id: 2,
              stable_id: "sid-2",
              filename: "demo-lora-2.safetensors",
              file_path: "/tmp/demo-lora-2.safetensors",
              base_model_code: "FLX",
              category_code: "STL",
              role: "style",
              has_block_weights: true,
              block_layout: "flux_fallback_16",
            },
          ],
          total: 2,
        });
      }

      if (url.endsWith("/lora/combine") && init?.method === "POST") {
        if (combineMode === "structured-error") {
          return jsonResponse(
            {
              detail: {
                response_schema_version: "7.1",
                compatible: false,
                validated_base_model: "FLX",
                validated_layout: "flux_fallback_16",
                included_loras: ["sid-1"],
                excluded_loras: ["sid-2"],
                reasons: ["sid-2 is incompatible with the selected stack"],
                warnings: ["Incompatible stack"],
                node_payloads: [],
              },
            },
            false,
            400
          );
        }

        return jsonResponse({
          response_schema_version: "7.1",
          compatible: true,
          validated_base_model: "FLX",
          validated_layout: "flux_fallback_16",
          included_loras: ["sid-1", "sid-2"],
          excluded_loras: [],
          reasons: [],
          warnings: [],
          combined: {
            strength_model: 1.0,
            strength_clip: null,
            block_weights: [0.85, 0.75, 0.65],
            block_weights_csv: "0.8500,0.7500,0.6500",
          },
          node_payloads: [
            {
              stable_id: "sid-1",
              filename: "demo-lora-1.safetensors",
              role: "style",
              base_model_code: "FLX",
              block_layout: "flux_fallback_16",
              strength_model: 0.8,
              strength_clip: 1.0,
              role_strength_recommendation: {
                recommended_model_strength: 0.45,
                recommended_clip_strength: 0.25,
                applied_to_math: false,
                basis: "role_policy_advisory",
              },
              block_weights: [1.0, 0.9, 0.8],
              block_weights_csv: "1.0000,0.9000,0.8000",
              orchestration_notes: ["Phase 8.5 smoke note for sid-1"],
            },
            {
              stable_id: "sid-2",
              filename: "demo-lora-2.safetensors",
              role: "style",
              base_model_code: "FLX",
              block_layout: "flux_fallback_16",
              strength_model: 0.6,
              strength_clip: 0.9,
              role_strength_recommendation: {
                recommended_model_strength: 0.35,
                recommended_clip_strength: 0.15,
                applied_to_math: false,
                basis: "role_policy_advisory",
              },
              block_weights: [0.7, 0.6, 0.5],
              block_weights_csv: "0.7000,0.6000,0.5000",
              orchestration_notes: ["Phase 8.5 smoke note for sid-2"],
            },
          ],
        });
      }

      return jsonResponse({}, false, 404);
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  function selectTwoLorasInCombine() {
    fireEvent.click(screen.getAllByRole("tab", { name: "Combine" })[0]);
    fireEvent.click(screen.getAllByText("sid-1")[0]);
    fireEvent.click(screen.getAllByText("sid-2")[0]);
  }

  it("selects multiple LoRAs and renders combine configuration", async () => {
    render(<App />);

    expect(await screen.findByText(/LoRA catalog/i)).toBeTruthy();

    await waitFor(() => expect(screen.getAllByText("sid-1").length).toBeGreaterThan(0));
    selectTwoLorasInCombine();

    fireEvent.click(screen.getByRole("button", { name: /calculate/i }));

    await waitFor(() => {
      const selectedStack = screen.getAllByText(/Selected stack/i)[0].closest("section");
      expect(selectedStack).toBeTruthy();

      const stack = within(selectedStack);
      expect(stack.getAllByText("sid-1").length).toBeGreaterThan(0);
      expect(stack.getAllByText("sid-2").length).toBeGreaterThan(0);
      expect(stack.getByText(/0\.45/)).toBeTruthy();
      expect(stack.getByText(/0\.25/)).toBeTruthy();
      expect(stack.getByText(/1\.0,0\.9,0\.8/i)).toBeTruthy();
      expect(stack.getByText(/0\.7,0\.6,0\.5/i)).toBeTruthy();
      expect(stack.getByText(/Stack Health/i)).toBeTruthy();
      expect(stack.getByText(/Softening notes/i)).toBeTruthy();
      expect(stack.getByText("Phase 8.5 smoke note for sid-1")).toBeTruthy();
      expect(stack.getByText("Phase 8.5 smoke note for sid-2")).toBeTruthy();
    });
  });

  it("renders stack health for structured combine errors", async () => {
    combineMode = "structured-error";
    render(<App />);

    expect(await screen.findByText(/LoRA catalog/i)).toBeTruthy();

    await waitFor(() => expect(screen.getAllByText("sid-1").length).toBeGreaterThan(0));
    selectTwoLorasInCombine();

    fireEvent.click(screen.getByRole("button", { name: /calculate/i }));

    await waitFor(() => {
      const health = screen.getByRole("region", { name: /Stack health/i });
      const healthPanel = within(health);

      expect(healthPanel.getByText(/Stack Health/i)).toBeTruthy();
      expect(healthPanel.getByText(/Needs attention/i)).toBeTruthy();
      expect(healthPanel.getByText(/Base/i)).toBeTruthy();
      expect(healthPanel.getByText(/FLX/i)).toBeTruthy();
      expect(healthPanel.getByText(/Excluded/i)).toBeTruthy();      expect(healthPanel.getByText(/Warnings/i)).toBeTruthy();

      expect(screen.getAllByText(/Incompatible stack/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/sid-2/i).length).toBeGreaterThan(0);
    });
  });
});











