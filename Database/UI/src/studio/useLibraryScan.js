import { useEffect, useRef, useState } from "react";

export function useLibraryScan(apiBase, onInventoryStart, onInventoryReady) {
  const [status, setStatus] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const callbacks = useRef({ onInventoryStart, onInventoryReady });
  useEffect(() => { callbacks.current = { onInventoryStart, onInventoryReady }; });
  const generation = useRef(0), alive = useRef(true), started = useRef(null), inventory = useRef(null);
  function accept(data) {
    if (!data || !["idle", "running", "complete", "partial", "cancelled", "failed"].includes(data.status)) throw new Error("Library scan status is unavailable.");
    if (data.status === "running" && data.phase === "catalogue" && started.current !== data.job_id) { started.current = data.job_id; callbacks.current.onInventoryStart(); }
    if (data.catalogue_scan_id && inventory.current !== data.catalogue_scan_id) { inventory.current = data.catalogue_scan_id; callbacks.current.onInventoryReady(data); }
    setStatus(data); setError("");
  }
  async function request(action) {
    const requestId = ++generation.current;
    if (action) setPending(true);
    try {
      const response = await fetch(`${apiBase}/library-scan${action && action !== "start" ? `/${action}` : ""}`, action ? { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" } : undefined);
      const data = await response.json();
      if (!alive.current || requestId !== generation.current) return;
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || `Library check unavailable (${response.status}).`);
      accept(data); return data;
    } catch (failure) { if (alive.current && requestId === generation.current) setError(failure.message); return { error: failure.message }; }
    finally { if (alive.current && requestId === generation.current) setPending(false); }
  }
  useEffect(() => { alive.current = true; request(); return () => { alive.current = false; generation.current += 1; }; /* Browser reads status; service owns startup scan. */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);
  useEffect(() => {
    if (status?.status !== "running" || pending || error) return;
    const timer = setTimeout(() => request(), 1000);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, pending, error]);
  return { status, error, pending, inventoryBusy: pending || status?.status === "running" && status.phase === "catalogue", refresh: () => request("start"), resume: () => request("resume"), cancel: () => request("cancel"), retry: () => request() };
}

