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

describe("App block weights smoke", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn(async (input) => {
      const url = String(input);
      if (url.includes("/lora/search")) {
        return jsonResponse({
          results: [
            {
              id: 1,
              stable_id: "sid-1",
              filename: "demo-lora.safetensors",
              file_path: "/tmp/demo-lora.safetensors",
              has_block_weights: true,
              block_layout: "flux_fallback_16",
            },
          ],
          total: 1,
        });
      }
      if (url.endsWith("/lora/sid-1")) {
        return jsonResponse({
          stable_id: "sid-1",
          filename: "demo-lora.safetensors",
          file_path: "/tmp/demo-lora.safetensors",
          lora_type: "unet",
        });
      }
      if (url.endsWith("/lora/sid-1/blocks")) {
        return jsonResponse({
          blocks: [
            { block_index: 0, weight: 0.0, raw_strength: null },
            { block_index: 1, weight: 0.0, raw_strength: null },
          ],
          fallback: false,
        });
      }
      if (url.endsWith("/lora/sid-1/profiles")) {
        return jsonResponse({ profiles: [] });
      }
      return jsonResponse({}, false, 404);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders block weights and updates value when bar is clicked", async () => {
    render(<App />);

    expect(await screen.findByText(/LoRA catalog/i)).toBeTruthy();

    fireEvent.click(await screen.findByText("demo-lora.safetensors"));

    const input = await screen.findByLabelText("Block 0 weight");
    const track = await screen.findByTestId("block-bar-track-0");

    track.getBoundingClientRect = () => ({
      left: 0,
      width: 100,
      top: 0,
      right: 100,
      bottom: 10,
      height: 10,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    fireEvent.pointerDown(track, { clientX: 50, pointerId: 1 });
    fireEvent.pointerUp(track, { clientX: 50, pointerId: 1 });

    await waitFor(() => {
      expect(input.value).toBe("0.5");
    });
  });
});
