import { useEffect, useRef, useState } from "react";

async function api(url, body) {
  const response = await fetch(url, body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(typeof data.detail === "string" ? data.detail : `Recipe request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

export default function CompositionPanel({ apiBase, selectedIds, versionIds, result, dirty, loading, onBusyChange, currentRecipe, onSaved, onRestore, onVersionChange, onInvalidatePrepared }) {
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const [name, setName] = useState(currentRecipe?.name || "");
  const [versions, setVersions] = useState([]);
  const [chosen, setChosen] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const missing = selectedIds.filter((id) => !versionIds[id]);
  const endpoint = `${apiBase}/composition-versions`;
  useEffect(() => {
    let active = true;
    api(endpoint).then((data) => { if (active) setVersions(data.versions || []); }).catch((failure) => { if (active) setError(failure.message); });
    return () => { active = false; };
  }, [endpoint]);
  async function capture() {
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    try {
      for (const id of missing) {
        const version = await api(`${apiBase}/profile-versions/${encodeURIComponent(id)}/defaults`, {});
        if (!alive.current) return;
        onVersionChange(id, version.version_id);
      }
      setMessage("Defaults captured. Prepare the stack again to bind the result to these exact versions.");
    } catch (failure) { setError(failure.message); }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }
  async function save() {
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    try {
      const saved = await api(endpoint, { name: name.trim(), ...(currentRecipe ? { parent_version_id: currentRecipe.version_id } : {}), target_contract_id: "flux1-dev-native-v1", entries: selectedIds.map((id) => ({ stable_id: id, profile_version_id: versionIds[id] })), expected_preparation_digest: result.preparation_digest });
      if (!alive.current) return;
      setVersions((previous) => [...previous, saved]);
      onSaved(saved);
      setMessage("Recipe saved as a new immutable version. Previous recipes are preserved.");
    } catch (failure) { if (alive.current) { if (failure.status === 409 || failure.status === 422) onInvalidatePrepared(); setError(failure.message); } }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }
  async function load() {
    setBusy(true); onBusyChange(true); setError("");
    try { const recipe = await api(`${endpoint}/${encodeURIComponent(chosen)}`); if (alive.current) onRestore(recipe); }
    catch (failure) { setError(failure.message); }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }
  return <section className="studio-panel studio-recipes" aria-label="Composition recipes"><div className="studio-section-heading"><div><span className="studio-eyebrow">Keep the whole composition</span><h2>Recipes</h2></div>{currentRecipe && <span className="studio-tag">{currentRecipe.name}</span>}</div>
    <p className="studio-help">A recipe keeps loader order, exact profile versions and the server result. Loading restores the stack; prepare it again to check the current files before copying.</p>
    <div className="studio-recipe-fields"><label>Recipe name<input disabled={busy || loading} value={name} maxLength={200} onChange={(event) => setName(event.target.value)} placeholder="e.g. Portrait in linen" /></label><button className="studio-primary" disabled={busy || loading || dirty || !name.trim() || !selectedIds.length || missing.length > 0 || !result?.preparation_digest || result.compatible === false} onClick={save}>Save recipe version</button></div>
    {missing.length > 0 && <div className="studio-recipe-capture"><p className="studio-help">{missing.length} LoRA{missing.length === 1 ? " needs" : "s need"} a captured Default before this recipe can be saved.</p><button disabled={busy || loading || dirty} onClick={capture}>Capture missing Defaults</button></div>}
    <div className="studio-recipe-fields"><label>Saved recipe<select aria-label="Saved recipe" disabled={busy || loading} value={chosen} onChange={(event) => setChosen(event.target.value)}><option value="">Choose a saved recipe…</option>{versions.map((version) => <option value={version.version_id} key={version.version_id}>{version.name} · {new Date(version.created_at).toLocaleString()}</option>)}</select></label><button onClick={load} disabled={!chosen || busy || loading || dirty}>Load recipe</button></div>
    {dirty && <p className="studio-help">Save or discard all personal drafts before saving or loading a recipe.</p>}
    {message && <p role="status">{message}</p>}{error && <p role="alert" className="studio-alert">{error}</p>}
  </section>;
}
