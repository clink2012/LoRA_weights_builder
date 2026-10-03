import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import LibraryScanStatus from "./LibraryScanStatus";
const scan = { status: { status: "complete", issue_count: 2, scanned: 10, remaining: 0, unchecked_family_count: 1, catalogue_scan_id: "one" }, error: "", pending: false, retry: vi.fn() };
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("library observations disclosure", () => {
  it("loads only on opening and keeps unsupported families distinct from potential header problems", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ total: 2, results: [{ stable_id: "unknown", filename: "New family", source_status: "current", observation: { issues: [{ severity: "not_checked", message: "This family has no header checker yet." }] } }, { stable_id: "moved", filename: "Changed file", source_status: "stale", observation: { issues: [{ severity: "potential_issue", message: "Folder label disagrees with header evidence." }] } }] }) });
    vi.stubGlobal("fetch", fetch); const { container } = render(<LibraryScanStatus apiBase="/api" scan={scan} />);
    expect(fetch).not.toHaveBeenCalled();
    fireEvent.click(container.querySelector("summary"));
    await screen.findByText("New family");
    expect(screen.getByText("Not verified")).toBeTruthy(); expect(screen.getByText("Potential issue")).toBeTruthy();
    expect(screen.getByText("Source changed; recheck required.")).toBeTruthy();
    expect(fetch.mock.calls[0][0]).toBe("/api/library-scan/issues?limit=20&offset=0");
  });
  it("returns observations to the first page when a new inventory replaces the previous scan", async () => {
    const fetch = vi.fn().mockImplementation(async () => ({ ok: true, json: async () => ({ total: 25, results: [] }) }));
    vi.stubGlobal("fetch", fetch);
    const { container, rerender } = render(<LibraryScanStatus apiBase="/api" scan={scan} />);
    fireEvent.click(container.querySelector("summary"));
    await screen.findByRole("button", { name: "More observations" });
    fireEvent.click(screen.getByRole("button", { name: "More observations" }));
    await waitFor(() => expect(fetch.mock.calls.at(-1)[0]).toContain("offset=20"));
    rerender(<LibraryScanStatus apiBase="/api" scan={{ ...scan, status: { ...scan.status, catalogue_scan_id: "two" } }} />);
    await waitFor(() => expect(fetch.mock.calls.at(-1)[0]).toContain("offset=0"));
  });
  it("does not represent idle or failed checks as a successful zero-issue scan", async () => {
    vi.stubGlobal("fetch", vi.fn()); const { rerender } = render(<LibraryScanStatus apiBase="/api" scan={{ ...scan, status: { status: "idle" } }} />);
    expect(screen.getByText("Library checks not started")).toBeTruthy();
    rerender(<LibraryScanStatus apiBase="/api" scan={{ ...scan, status: { status: "failed" }, error: "Root unavailable" }} />);
    await waitFor(() => expect(screen.getByText("Library checks need attention")).toBeTruthy());
  });
});
