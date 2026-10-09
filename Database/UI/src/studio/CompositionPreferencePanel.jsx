import { useEffect, useRef, useState } from "react";

const TARGET = "flux1-dev-native-v1";
async function request(url, body) {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(typeof data.detail === "string" ? data.detail : `Saved preference request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function validateRecall(data, ids) {
  if (!data || !["none", "preferred", "needs_review"].includes(data.status) || data.target_contract_id !== TARGET || JSON.stringify(data.stable_ids) !== JSON.stringify(ids)) throw new Error("The preferred recipe response does not match this ordered combination.");
  if (data.status === "preferred" && (!data.recipe?.version_id || data.recipe.target_contract_id !== TARGET || !Array.isArray(data.recipe.entries) || data.recipe.entries.length !== ids.length || data.recipe.entries.some((entry, index) => entry.stable_id !== ids[index] || typeof entry.profile_version_id !== "string"))) throw new Error("The preferred recipe response is incomplete.");
  return data;
}

export default function CompositionPreferencePanel({ apiBase, selectedIds, versionIds, currentRecipe, result, dirty, loading, onBusyChange, onRestore, onInvalidatePrepared }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const endpoint = `${apiBase}/composition-preferences`;
  const selectionKey = JSON.stringify(selectedIds);
  const versionKey = JSON.stringify(selectedIds.map((id) => versionIds[id] || null));
  const latest = useRef(null);
  const alive = useRef(false);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    const previous = latest.current;
    const changed = !previous || previous.selectionKey !== selectionKey || previous.versionKey !== versionKey || previous.dirty !== dirty || previous.loading !== loading || previous.currentRecipe?.version_id !== currentRecipe?.version_id;
    latest.current = { selectionKey, versionKey, dirty, loading, currentRecipe, revision: (previous?.revision || 0) + (changed ? 1 : 0) };
  });
  async function loadPreferred() {
    const start = latest.current;
    const ids = JSON.parse(selectionKey);
    if (!ids.length || start.dirty || start.loading || busy) return;
    setBusy(true); onBusyChange(true); setError(''); setMessage('');
    try {
      const raw = await request(`${endpoint}/resolve`, { stable_ids: ids, target_contract_id: TARGET });
      if (!alive.current) return;
      const data = validateRecall(raw, ids);
      const now = latest.current;
      if (now.selectionKey !== selectionKey || now.versionKey !== start.versionKey || now.dirty || now.currentRecipe?.version_id !== start.currentRecipe?.version_id) return;
      if (data.status === "preferred") onRestore({ ...data.recipe, recalled_preference: true });
      else if (data.status === "needs_review") setMessage(data.reason || "Your preferred recipe needs review against the current files.");
      else setMessage("No preferred recipe is saved for this ordered combination. Choose a recipe from Saved compositions.");
    } catch (failure) {
      if (alive.current) setError(`Preferred recipe could not be checked: ${failure.message}`);
    } finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }

  const recipeMatches = currentRecipe?.version_id && Array.isArray(currentRecipe.entries) && currentRecipe.entries.length === selectedIds.length && currentRecipe.entries.every((entry, index) => entry.stable_id === selectedIds[index] && entry.profile_version_id === versionIds[entry.stable_id]);

  async function choose() {
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    try {
      const data = validateRecall(await request(`${endpoint}/choose`, { version_id: currentRecipe.version_id, expected_preparation_digest: result.preparation_digest }), selectedIds);
      if (data.status !== "preferred" || data.recipe.version_id !== currentRecipe.version_id) throw new Error("The saved preference response does not match the chosen recipe.");
      if (alive.current) setMessage("This recipe is preferred for this ordered combination. Use Load preferred recipe to choose it next time.");
    } catch (failure) { if (alive.current) { if (failure.status === 409 || failure.status === 422) onInvalidatePrepared(); setError(failure.message); } }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }

  async function originals() {
    setBusy(true); onBusyChange(true); setError(""); setMessage(""); onInvalidatePrepared();
    try {
      const data = await request(`${endpoint}/originals`, { stable_ids: selectedIds, target_contract_id: TARGET });
      if (data.status !== "originals" || data.target_contract_id !== TARGET || JSON.stringify(data.stable_ids) !== selectionKey || !Array.isArray(data.entries) || data.entries.length !== selectedIds.length || data.entries.some((entry, index) => entry.stable_id !== selectedIds[index] || typeof entry.profile_version_id !== "string")) throw new Error("The original values response does not match this ordered combination.");
      if (alive.current) onRestore(data);
    } catch (failure) { if (alive.current) setError(failure.message); }
    finally { if (alive.current) { setBusy(false); onBusyChange(false); } }
  }

  if (!selectedIds.length) return null;
  return <div className="studio-composition-preference">
    <p className="studio-help">Saved recipes load only when you choose them. A preferred recipe is a shortcut for this exact loader order; it never replaces a new starting proposal automatically.</p>
    <div className="studio-recipe-fields"><button disabled={busy || loading || dirty} onClick={loadPreferred}>Load preferred recipe</button><button disabled={busy || loading || dirty || !recipeMatches || !result?.preparation_digest || result.compatible === false} onClick={choose}>Make this recipe preferred</button><button disabled={busy || loading || dirty} onClick={originals}>Return combination to original values</button></div>
    {currentRecipe?.recalled_preference && <p role="status">Preferred recipe loaded: {currentRecipe.name}. Prepare again before copying.</p>}
    {message && <p role="status">{message}</p>}{error && <p role="alert">{error}</p>}
  </div>;
}
