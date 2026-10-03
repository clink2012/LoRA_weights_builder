import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import RuntimeBadge from "./RuntimeBadge";

const status = (database_kind) => ({ app: "lora-comfy-combiner-local", host: "127.0.0.1", database_kind, database_path: "E:/private/main.db", build: { source_revision: "12345678abcdef" } });
const response = (data, ok = true) => ({ ok, json: async () => data });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("local runtime identity", () => {
  it.each([["copy", "Preview database"], ["main", "Main database"]])("uses the launcher's %s identity without displaying paths", async (kind, label) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(status(kind))));
    render(<RuntimeBadge development={false} />);
    expect(await screen.findByText(label)).toBeTruthy();
    expect(document.body.textContent).not.toContain("E:/private");
    expect(screen.getByText("12345678")).toBeTruthy();
  });
  it.each([{ ...status("main"), app: "other-app" }, { ...status("main"), database_kind: "unknown" }, { ...status("main"), host: "0.0.0.0" }])("does not infer main or preview from an unrecognised response", async (payload) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(payload)));
    render(<RuntimeBadge development={false} />);
    expect(await screen.findByText("Database identity unavailable")).toBeTruthy();
    expect(screen.queryByText("Main database")).toBeNull();
  });
  it("labels an unconfirmed development server truthfully and supports retry", async () => {
    const request = vi.fn().mockResolvedValueOnce(response({}, false)).mockResolvedValueOnce(response(status("copy")));
    vi.stubGlobal("fetch", request);
    render(<RuntimeBadge development />);
    expect(await screen.findByText("Development · database unverified")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Runtime: Development · database unverified"));
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    await waitFor(() => expect(screen.getByText("Preview database")).toBeTruthy());
    expect(request).toHaveBeenLastCalledWith("/local-app/status", expect.objectContaining({ cache: "no-store" }));
  });
});
