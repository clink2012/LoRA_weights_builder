import { useEffect, useRef, useState } from "react";

async function request(url, body) {
  const response = await fetch(url, body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Profile request failed (${response.status})`);
  return data;
}

export function useProfileVariants(apiBase, versionIds, onVersionChange, onDraftChange) {
  const [records, setRecords] = useState({});
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const patch = (id, update) => setRecords((previous) => ({ ...previous, [id]: { ...previous[id], ...update } }));
  const endpoint = (id) => `${apiBase}/profile-versions/${encodeURIComponent(id)}`;

  async function open(id) {
    patch(id, { busy: true, error: "" });
    try {
      const root = await request(`${endpoint(id)}/defaults`, {});
      const [history, selected] = await Promise.all([
        request(`${endpoint(id)}?default_id=${encodeURIComponent(root.version_id)}`),
        versionIds[id] ? request(`${endpoint(id)}/versions/${encodeURIComponent(versionIds[id])}`) : request(`${endpoint(id)}/selection?default_id=${encodeURIComponent(root.version_id)}`),
      ]);
      if (!alive.current) return;
      if (selected.default_id !== root.version_id) throw new Error("This selection belongs to an earlier source or loader contract. Its history is retained, but it cannot be edited against the current Default.");
      patch(id, { root, versions: history.versions, selected, draft: null, name: "", busy: false });
      onVersionChange(id, selected.version_id);
    } catch (error) { patch(id, { busy: false, error: error.message }); }
  }

  function edit(id, update) {
    const record = records[id];
    if (record.busy) return;
    const current = record.draft || { values: [...record.selected.values], settings: { ...record.selected.settings }, ab: structuredClone(record.selected.ab || {}) };
    patch(id, { draft: { ...current, ...update }, error: "" });
    onDraftChange(id, true);
  }

  function editValue(id, index, value) {
    const record = records[id];
    if (record.busy) return;
    const current = record.draft || record.selected;
    const values = [...current.values];
    values[index] = value;
    // Editing a resolved A/B slot removes that experiment until explicitly reset.
    const label = record.selected.binding.slots[index].label;
    const ab = Object.fromEntries(Object.entries(current.ab || {}).filter(([, experiment]) => !experiment.slot_labels.includes(label)));
    edit(id, { values, ab });
  }

  function discard(id) { patch(id, { draft: null, name: "", error: "" }); onDraftChange(id, false); }

  async function select(id, versionId) {
    const record = records[id];
    if (record.draft) return;
    patch(id, { busy: true, error: "" });
    try {
      const selected = await request(`${endpoint(id)}/selection`, { default_id: record.root.version_id, version_id: versionId });
      if (!alive.current) return;
      patch(id, { selected, busy: false, draft: null, name: "" });
      onVersionChange(id, selected.version_id);
    } catch (error) { patch(id, { busy: false, error: error.message }); }
  }

  async function save(id) {
    const record = records[id];
    if (!record.draft) return;
    patch(id, { busy: true, error: "" });
    try {
      const saved = await request(`${endpoint(id)}/revisions`, { default_id: record.root.version_id, parent_id: record.selected.version_id, name: record.name.trim(), ...record.draft });
      if (!alive.current) return;
      patch(id, { selected: saved, versions: [...record.versions, saved], draft: null, name: "", busy: true });
      onDraftChange(id, false);
      onVersionChange(id, saved.version_id);
      try { await request(`${endpoint(id)}/selection`, { default_id: record.root.version_id, version_id: saved.version_id }); }
      catch { patch(id, { error: "Your revision was saved. Its active selection could not be remembered; choose it from history to retry." }); }
      finally { if (alive.current) patch(id, { busy: false }); }
    } catch (error) { patch(id, { busy: false, error: error.message }); }
  }

  return { records, open, edit, editValue, discard, select, save, setName: (id, name) => patch(id, { name }) };
}
