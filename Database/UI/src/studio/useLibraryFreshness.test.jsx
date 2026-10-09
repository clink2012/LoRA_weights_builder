import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useLibraryFreshness } from "./useLibraryFreshness";
const data = (status = "current", id = "one") => ({ status, counts: { added: 0, removed: 0, changed: 0, returned: 0 }, catalogue_scan_id: id, checked_at: "2026-10-09T12:00:00Z" });
const response = (value, ok = true) => ({ ok, json: async () => value });
function Harness({ id, outdated }) {
  const check = useLibraryFreshness("/api", id, outdated);
  return <><span>{check.checking ? "Checking" : check.error || check.data.status}</span><button onClick={check.check}>Recheck</button></>;
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
describe("library reopening comparison", () => {
  it("checks on browser opening and manually without starting a catalogue mutation", async () => {
    const fetch = vi.fn().mockResolvedValue(response(data("outdated"))), outdated = vi.fn();
    vi.stubGlobal("fetch", fetch); render(<Harness id="one" outdated={outdated} />);
    await screen.findByText("outdated"); expect(outdated).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0]).toEqual(["/api/library-scan/freshness"]);
    fetch.mockResolvedValue(response(data())); fireEvent.click(screen.getByRole("button"));
    await screen.findByText("current"); expect(outdated).toHaveBeenCalledTimes(1);
  });
  it("rechecks after inventory replacement and ignores the older answer", async () => {
    let finish;
    vi.stubGlobal("fetch", vi.fn().mockReturnValueOnce(new Promise((resolve) => { finish = resolve; })).mockResolvedValue(response(data("current", "two"))));
    const outdated = vi.fn(), { rerender } = render(<Harness id="one" outdated={outdated} />);
    rerender(<Harness id="two" outdated={outdated} />); await screen.findByText("current");
    await act(async () => finish(response(data("outdated"))));
    expect(screen.getByText("current")).toBeTruthy(); expect(outdated).not.toHaveBeenCalled();
  });
  it("failure or malformed data is never a matching-inventory claim", async () => {
    const fetch = vi.fn().mockResolvedValue(response({ detail: { reason: "Folder offline" } }, false));
    vi.stubGlobal("fetch", fetch); render(<Harness id="one" />);
    await screen.findByText("Folder offline");
    fetch.mockResolvedValue(response({ status: "current" })); fireEvent.click(screen.getByRole("button"));
    await screen.findByText("The library folder check is unavailable.");
  });
});
