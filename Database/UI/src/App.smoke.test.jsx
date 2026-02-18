import { render, screen, fireEvent, waitFor } from "@testing-library/react";
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
  beforeEach(() => {
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
              has_block_weights: true,
              block_layout: "flux_fallback_16",
            },
            {
              id: 2,
              stable_id: "sid-2",
              filename: "demo-lora-2.safetensors",
              file_path: "/tmp/demo-lora-2.safetensors",
              has_block_weights: true,
              block_layout: "flux_fallback_16",
            },
          ],
          total: 2,
        });
      }

      if (url.endsWith("/lora/combine") && init?.method === "POST") {
        return jsonResponse({
          validated_base_model: "FLX",
          validated_layout: "flux_fallback_16",
          warnings: [],
          excluded_loras: [],
          combined: {
            "sid-1": {
              strength_model: 0.8,
              strength_clip: 1,
              block_weights: [1, 0.9, 0.8],
            },
            "sid-2": {
              strength_model: 0.6,
              strength_clip: 0.9,
              block_weights: [0.7, 0.6, 0.5],
            },
          },
        });
      }

      return jsonResponse({}, false, 404);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("selects multiple LoRAs and renders combine configuration", async () => {
    render(<App />);

    expect(await screen.findByText(/LoRA catalog/i)).toBeTruthy();

    fireEvent.click(await screen.findByLabelText("Select sid-1"));
    fireEvent.click(await screen.findByLabelText("Select sid-2"));

    fireEvent.click(screen.getByRole("tab", { name: "Combine" }));
    fireEvent.click(screen.getByRole("button", { name: "Calculate configuration" }));

    await waitFor(() => {
      expect(screen.getAllByText("sid-1").length).toBeGreaterThan(0);
      expect(screen.getAllByText("sid-2").length).toBeGreaterThan(0);
      expect(screen.getByText(/strength_model: 0.8/i)).toBeTruthy();
    });
  });
});
