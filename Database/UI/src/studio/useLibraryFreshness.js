import { useEffect, useRef, useState } from "react";

// Opening the browser compares filenames/stat only. Updating the catalogue
// and header observations remains an explicit scan owned by the service.
export function useLibraryFreshness(apiBase, inventoryId, onOutdated) {
  const [state, setState] = useState({ data: null, error: "", checking: true });
  const [retry, setRetry] = useState(0);
  const key = `${apiBase}:${inventoryId || "unscanned"}:${retry}`;
  const callback = useRef(onOutdated);
  useEffect(() => { callback.current = onOutdated; });
  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/library-scan/freshness`).then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || "The library folder could not be checked.");
      if (!["current", "outdated", "not_scanned"].includes(data.status) || !data.counts || !data.checked_at) throw new Error("The library folder check is unavailable.");
      if (active) {
        setState({ key, data, error: "", checking: false });
        if (data.status !== "current") callback.current?.();
      }
    }).catch((failure) => { if (active) setState({ key, data: null, error: failure.message, checking: false }); });
    return () => { active = false; };
  }, [apiBase, inventoryId, retry, key]);
  return { ...(state.key === key ? state : { data: null, error: "", checking: true }), check: () => setRetry((value) => value + 1) };
}
