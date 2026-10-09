import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import LibraryLocation from "./LibraryLocation";
const response = (data, ok = true) => ({ ok, json: async () => data });
const original = { active_root: "E:/loras", selected_root: "E:/loras", restart_required: false };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("library location selection", () => {
  it("loads on opening and saves the next-start folder without claiming a live switch", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response(original)).mockResolvedValue(response({ ...original, selected_root: "F:/loras", restart_required: true }));
    vi.stubGlobal("fetch", fetch); const changed = vi.fn(); render(<LibraryLocation apiBase="/api" onChanged={changed} />);
    expect(fetch).not.toHaveBeenCalled(); fireEvent.click(screen.getByText("Library folder"));
    fireEvent.change(await screen.findByLabelText("LoRA folder path"), { target: { value: "F:/loras" } });
    fireEvent.click(screen.getByRole("button"));
    await screen.findByText(/Restart the app to use it/);
    expect(screen.getByText("Current folder: E:/loras")).toBeTruthy(); expect(changed).toHaveBeenCalledOnce();
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ root: "F:/loras" });
  });
  it("preserves the active folder and edited input when saving is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(response(original)).mockResolvedValue(response({ detail: { reason: "Folder unavailable" } }, false)));
    const changed = vi.fn(); render(<LibraryLocation apiBase="/api" onChanged={changed} />);
    fireEvent.click(screen.getByText("Library folder"));
    fireEvent.change(await screen.findByLabelText("LoRA folder path"), { target: { value: "F:/bad" } }); fireEvent.click(screen.getByRole("button"));
    await screen.findByText("Folder unavailable"); expect(screen.getByLabelText("LoRA folder path").value).toBe("F:/bad"); expect(changed).not.toHaveBeenCalled();
  });
});
