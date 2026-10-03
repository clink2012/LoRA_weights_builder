import { useEffect, useState } from "react";

const APP_ID = "lora-comfy-combiner-local";

export default function RuntimeBadge({ development = import.meta.env.DEV }) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState({ kind: "checking" });
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 5000);
    async function inspect() {
      try {
        const response = await fetch("/local-app/status", { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error("Runtime identity is unavailable");
        const status = await response.json();
        if (status?.app !== APP_ID || status.host !== "127.0.0.1" || !["copy", "main"].includes(status.database_kind)) throw new Error("Runtime identity was not recognised");
        if (active) setState({ kind: status.database_kind, revision: typeof status.build?.source_revision === "string" ? status.build.source_revision.slice(0, 8) : null });
      } catch {
        if (active) setState({ kind: "unavailable" });
      } finally { clearTimeout(deadline); }
    }
    inspect();
    return () => { active = false; clearTimeout(deadline); controller.abort(); };
  }, [attempt]);
  const label = state.kind === "copy" ? "Preview database" : state.kind === "main" ? "Main database" : state.kind === "checking" ? "Checking database…" : development ? "Development · database unverified" : "Database identity unavailable";
  const detail = state.kind === "copy" ? "Changes in this session use a separate database copy." : state.kind === "main" ? "Changes in this session are saved to your main database." : state.kind === "checking" ? "Reading the identity of this local app session." : "The local launcher has not confirmed which database this page is using.";
  return <details className={`studio-runtime studio-runtime-${state.kind}`}><summary aria-label={`Runtime: ${label}`}><span className="studio-runtime-dot" aria-hidden="true" /><span role="status">{label}</span></summary><div className="studio-runtime-details"><p>{detail}</p>{state.revision && <p>Built revision <code>{state.revision}</code></p>}<button disabled={state.kind === "checking"} onClick={() => { setState({ kind: "checking" }); setAttempt((previous) => previous + 1); }}>Check again</button></div></details>;
}
