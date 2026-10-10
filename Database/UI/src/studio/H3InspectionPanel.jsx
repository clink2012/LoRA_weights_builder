import { useEffect, useState } from "react";
import "./H3InspectionPanel.css";

export default function H3InspectionPanel({ apiBase, item, onClose }) {
  const [state, setState] = useState({ busy: true, result: null, error: "" });
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    fetch(`${apiBase}/h3-inspection/${encodeURIComponent(item.stable_id)}`, { signal: controller.signal })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "H3 inspection is unavailable.");
        if (active) setState({ busy: false, result: data, error: "" });
      }).catch((error) => {
        if (active && error.name !== "AbortError") setState({ busy: false, result: null, error: error.message });
      });
    return () => { active = false; controller.abort(); };
  }, [apiBase, item.stable_id]);
  const data = state.result;
  const max = Math.max(1, ...(data?.slots || []).map((slot) => slot.pair_count));
  return <section className="studio-panel h3-inspection" aria-label="H3 adapter inspection">
    <div className="studio-section-heading"><div><span className="studio-eyebrow">MiniMax H3 · adapter inspection</span><h2>{item.filename?.replace(/\.safetensors$/i, "") || item.stable_id}</h2></div><button className="studio-text-button" onClick={onClose}>Close inspection</button></div>
    <p className="studio-help">H3 block-weight preparation is in development. This view reads the adapter header; it does not load model tensors or prepare a ComfyUI export.</p>
    {state.busy && <p role="status">Reading H3 adapter coverage…</p>}
    {state.error && <p role="alert">{state.error}</p>}
    {data?.status === "unavailable" && <p role="alert">{data.reason}</p>}
    {data && data.status !== "unavailable" && <>
      <p role="status">{data.pair_count} ordinary pairs · {data.accounted_tensor_count} of {data.tensor_count} tensors accounted for · {data.issue_count} issues to review</p>
      <h3>Observed block coverage</h3>
      <p className="studio-help">Bar height counts complete low-rank pairs in each observed group. These are not block weights or measured contributions. Unshown indices have no observed complete pairs; the total model depth is not established.</p>
      {!!data.slots?.length && <div className="h3-coverage-chart" role="img" aria-label="H3 observed pair counts by block">
        {data.slots.map((slot) => <div className={`h3-coverage-slot h3-${slot.group}`} key={slot.label} title={`${slot.label}: ${slot.pair_count} pairs; ranks ${slot.ranks.join(", ")}`}><span>{slot.pair_count}</span><i style={{ height: `${slot.pair_count / max * 100}px` }} /><small>{slot.label}</small></div>)}
      </div>}
      <div className="h3-coverage-table"><table><caption>Observed groups and pair ranks</caption><thead><tr><th>Group / block</th><th>Pairs</th><th>Ranks</th></tr></thead><tbody>{(data.slots || []).map((slot) => <tr key={slot.label}><th scope="row">{slot.label}</th><td>{slot.pair_count}</td><td>{slot.ranks.join(", ")}</td></tr>)}</tbody></table></div>
      {!!data.issue_count && <><h3>Coverage needs review</h3><ul>{data.issues.map((issue, index) => <li key={index}><strong>{issue.code}</strong>: {issue.reason} <code>{issue.module}</code></li>)}</ul>{data.issue_count > data.issues.length && <p className="studio-help">Showing {data.issues.length} of {data.issue_count} issues.</p>}</>}
      <p className="studio-help">Architecture, checkpoint variant, target mapping, tensor strength and loader export remain unverified. Folder labels and observed names do not prove H3 compatibility. Required distillation/control adapters need separate handling before any balancing.</p>
      <p className="studio-help">Header fingerprint: <code>{data.file_identity?.header_sha256}</code> · Tensor payload read: {data.file_identity?.tensor_payload_read ? "yes" : "no"}</p>
    </>}
  </section>;
}
