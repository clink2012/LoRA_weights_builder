import { useEffect, useState } from "react";

export default function LibraryLocation({ apiBase, onChanged, busy }) {
  const [location, setLocation] = useState(null), [root, setRoot] = useState("");
  const [error, setError] = useState(""), [saving, setSaving] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    let active = true;
    fetch(`${apiBase}/library-scan/location`).then(async (response) => {
      const data = await response.json();
      if (!response.ok || !data.active_root || !data.selected_root) throw new Error("Library folder settings are unavailable.");
      if (active) { setLocation(data); setRoot(data.selected_root); }
    }).catch((failure) => { if (active) setError(failure.message); });
    return () => { active = false; };
  }, [apiBase, open]);
  async function save(event) {
    event.preventDefault(); setSaving(true); setError("");
    try {
      const response = await fetch(`${apiBase}/library-scan/location`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ root }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || "This library folder could not be saved.");
      if (!data.active_root || !data.selected_root) throw new Error("The saved folder setting could not be confirmed.");
      setLocation(data); setRoot(data.selected_root); onChanged?.();
    } catch (failure) { setError(failure.message); }
    finally { setSaving(false); }
  }
  return <details className="studio-scan-status" onToggle={(event) => setOpen(event.currentTarget.open)}><summary>Library folder</summary><div className="studio-scan-body">
    {location && <><p>Current folder: {location.active_root}</p>
      {location.restart_required && <p role="status">Folder saved: {location.selected_root}. Restart the app to use it; this session still uses the current folder.</p>}
      <form onSubmit={save}><label htmlFor="library-root">LoRA folder path</label><input id="library-root" value={root} onChange={(event) => setRoot(event.target.value)} disabled={saving || busy} /><button disabled={saving || busy || !root.trim()}>{saving ? "Saving folder…" : "Save folder for next start"}</button></form>
      <p>Use a full path to an existing folder. Changing folders preserves saved history. Files in a new location are discovered separately; matching names do not automatically reconnect old profiles.</p></>}
    {error && <p>{error}</p>}
  </div></details>;
}
