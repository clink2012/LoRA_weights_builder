import { useEffect, useRef, useState } from "react";

async function request(url, body) {
  const response = await fetch(url, body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Profile request failed (${response.status})`);
  return data;
}

export function useProfileVariants(apiBase, versionIds, onVersionChange, onDraftChange, prepared) {
  const [records, setRecords] = useState({});
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const patch = (id, update) => setRecords((previous) => ({ ...previous, [id]: { ...previous[id], ...update } }));
  const endpoint = (id) => `${apiBase}/profile-versions/${encodeURIComponent(id)}`;
  const [seenProposal, setSeenProposal] = useState(null);
  if (prepared?.starting_proposal && prepared !== seenProposal) {
    setSeenProposal(prepared);
    setRecords((previous) => Object.fromEntries(prepared.source_profiles.map((root, index) => {
      const values = prepared.starting_proposal.policy_preview.entries[index].values;
      return [root.stable_id, { root, versions: previous[root.stable_id]?.root?.version_id === root.version_id ? previous[root.stable_id].versions : [root],
        selected: root, computed: { ...root, name: 'Role-aware starting proposal', values: [...values], ab: {} }, draft: null, name: '', busy: false, error: '' }];
    })));
  }

  async function open(id, initialEdit) {
    patch(id, { busy: true, error: "" });
    try {
      const root = await request(`${endpoint(id)}/defaults`, {});
      const [history, selected] = await Promise.all([
        request(`${endpoint(id)}?default_id=${encodeURIComponent(root.version_id)}`),
        versionIds[id] ? request(`${endpoint(id)}/versions/${encodeURIComponent(versionIds[id])}`) : Promise.resolve(root),
      ]);
      if (!alive.current) return;
      if (selected.default_id !== root.version_id) throw new Error("This selection belongs to an earlier source or loader contract. Its history is retained, but it cannot be edited against the current Default.");
      let draft = null;
      if (initialEdit && Number.isInteger(initialEdit.index) && initialEdit.index >= 0 && initialEdit.index < selected.values.length && Number.isFinite(initialEdit.value)) {
        const values = [...selected.values];
        values[initialEdit.index] = initialEdit.value;
        const label = selected.binding.slots[initialEdit.index].label;
        const ab = Object.fromEntries(Object.entries(selected.ab || {}).filter(([, experiment]) => !experiment.slot_labels.includes(label)));
        draft = { values, settings: { ...selected.settings }, ab };
      }
      const computed = records[id]?.computed && selected.version_id === records[id].selected.version_id ? records[id].computed : null;
      patch(id, { root, versions: history.versions, selected, computed, draft, name: "", busy: false });
      onVersionChange(id, selected.version_id, false);
      if (draft) onDraftChange(id, true);
    } catch (error) { patch(id, { busy: false, error: error.message }); }
  }

  function edit(id, update) {
    const record = records[id];
    if (record.busy) return;
    setRecords((previous) => {
      const latest = previous[id];
      if (!latest || latest.busy) return previous;
      const base = latest.computed || latest.selected;
      const current = latest.draft || { values: [...base.values], settings: { ...base.settings }, ab: structuredClone(base.ab || {}) };
      const changed = { ...current, ...(typeof update === "function" ? update(current) : update) };
      if (latest.computed && !latest.draft) {
        // Functional updates retain all crossed bars within one pointer event.
        return Object.fromEntries(Object.entries(previous).map(([key, peer]) => [key,
          peer.computed ? { ...peer, computed: null, draft: key === id ? changed : {
            values: [...peer.computed.values], settings: { ...peer.computed.settings }, ab: structuredClone(peer.computed.ab || {}) }, error: "" } : peer]));
      }
      return { ...previous, [id]: { ...latest, draft: changed, error: "" } };
    });
    if (record.computed && !record.draft) {
      for (const [key, peer] of Object.entries(records)) if (peer.computed) onDraftChange(key, true);
    } else onDraftChange(id, true);
  }

  function editValue(id, index, value) {
    edit(id, (current) => {
      const values = [...current.values];
      values[index] = value;
      // Editing a resolved A/B slot removes that experiment until reset.
      const label = records[id].selected.binding.slots[index].label;
      const ab = Object.fromEntries(Object.entries(current.ab || {}).filter(([, experiment]) => !experiment.slot_labels.includes(label)));
      return { values, ab };
    });
  }

  function discard(id) { patch(id, { draft: null, name: "", error: "" }); onDraftChange(id, false); }

  async function select(id, versionId) {
    const record = records[id];
    if (record.draft) return;
    patch(id, { busy: true, error: "" });
    try {
      const selected = await request(`${endpoint(id)}/selection`, { default_id: record.root.version_id, version_id: versionId });
      if (!alive.current) return;
      if (record.computed) {
        setRecords((previous) => Object.fromEntries(Object.entries(previous).map(([key, peer]) => [key,
          key === id ? { ...peer, selected, computed: null, busy: false, draft: null, name: '' } :
          peer.computed ? { ...peer, computed: null, draft: { values: [...peer.computed.values], settings: { ...peer.computed.settings }, ab: structuredClone(peer.computed.ab || {}) } } : peer])));
        for (const [key, peer] of Object.entries(records)) if (key !== id && peer.computed) onDraftChange(key, true);
      } else patch(id, { selected, computed: null, busy: false, draft: null, name: "" });
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
      patch(id, { selected: saved, computed: null, versions: [...record.versions, saved], draft: null, name: "", busy: true });
      onDraftChange(id, false);
      onVersionChange(id, saved.version_id);
      try { await request(`${endpoint(id)}/selection`, { default_id: record.root.version_id, version_id: saved.version_id }); }
      catch { patch(id, { error: "Your revision was saved. Its active selection could not be remembered; choose it from history to retry." }); }
      finally { if (alive.current) patch(id, { busy: false }); }
    } catch (error) { patch(id, { busy: false, error: error.message }); }
  }

  return { records, open, edit, editValue, discard, select, save, setName: (id, name) => patch(id, { name }) };
}
