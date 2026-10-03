import { useState } from "react";

function localThumbnailUrl(apiBase, stableId) {
  if (!stableId) return null;
  try {
    const base = new URL(apiBase, window.location.href);
    if (base.origin !== window.location.origin || !["http:", "https:"].includes(base.protocol)) return null;
    return `${base.pathname.replace(/\/$/, "")}/lora/${encodeURIComponent(stableId)}/thumbnail`;
  } catch { return null; }
}

export default function LoraThumbnail({ apiBase, stableId, name }) {
  const [failed, setFailed] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const source = localThumbnailUrl(apiBase, stableId);
  return <span className="studio-thumbnail" aria-hidden="true">
    {!loaded && <span>{name.slice(0, 2).toUpperCase()}</span>}
    {source && !failed && <img src={source} alt="" width="44" height="52" loading="lazy" decoding="async" onLoad={() => setLoaded(true)} onError={() => { setFailed(true); setLoaded(false); }} />}
  </span>;
}
