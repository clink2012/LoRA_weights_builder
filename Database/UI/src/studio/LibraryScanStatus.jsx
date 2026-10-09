import { useEffect, useState } from "react";
import LibraryLocation from "./LibraryLocation";

export default function LibraryScanStatus({ apiBase, scan }) {
  const [open, setOpen] = useState(false), [issues, setIssues] = useState(null), [error, setError] = useState(""), [pageSelection, setPageSelection] = useState({ scanId: null, page: 0 });
  const status = scan.status;
  const page = pageSelection.scanId === status?.catalogue_scan_id ? pageSelection.page : 0;
  const setPage = (next) => setPageSelection({ scanId: status?.catalogue_scan_id, page: next });
  const issueKey = `${page}:${status?.catalogue_scan_id}:${status?.issue_count}`;
  const summary = scan.error ? "Library checks need attention" : !status ? "Library checks unavailable" : status.status === "idle" ? "Library checks not started" : status.status === "failed" ? "Library checks failed" : status.status === "running" ? status.phase === "catalogue" ? "Refreshing inventory…" : `Checking file headers · ${status.scanned} / ${status.total}` : `Library checks${status.status === "partial" ? " paused" : status.status === "cancelled" ? " stopped" : ""} · ${status.issue_count ?? 0} files to review`;
  const visibleIssues = issues?.key === issueKey ? issues.data : null;
  useEffect(() => {
    if (!open) return;
    let current = true;
    fetch(`${apiBase}/library-scan/issues?limit=20&offset=${page * 20}`).then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error("Could not load file observations."); if (current) { setIssues({ key: issueKey, data }); setError(""); } }).catch((failure) => { if (current) setError(failure.message); });
    return () => { current = false; };
  }, [apiBase, open, page, status?.catalogue_scan_id, status?.issue_count, issueKey]);
  const freshness = scan.freshness;
  const check = freshness?.data;
  const folderSummary = freshness?.checking ? "Checking for library changes…" : freshness?.error ? "Library folder check unavailable" : check?.status === "outdated" ? "Library database is out of date" : check?.status === "not_scanned" ? "Initial library scan needed" : check?.status === "current" ? "Library matches saved inventory" : null;
  return <><div className="studio-library-freshness" role="status">
    {folderSummary && <strong>{folderSummary}</strong>}
    {check && <><span> · {check.counts.added} added · {check.counts.removed} removed · {check.counts.changed} changed · {check.counts.returned} returned</span><small>Folder: {check.root} · Checked {new Date(check.checked_at).toLocaleString()} · File details only; model contents and compatibility are checked separately.</small></>}
    {freshness?.error && <p>{freshness.error}</p>}
    {freshness && <button onClick={freshness.check} disabled={freshness.checking || scan.inventoryBusy}>Check folder changes</button>}
    {check && check.status !== "current" && <button onClick={scan.refresh} disabled={scan.inventoryBusy}>Scan and update library</button>}
  </div><details className="studio-scan-status" onToggle={(event) => setOpen(event.currentTarget.open)}><summary>{summary}</summary><div className="studio-scan-body">
    {status && <><p>{status.message || "Local file observations only."}</p><p>{status.scanned ?? 0} checked · {status.remaining ?? 0} remaining · {status.unchecked_family_count ?? 0} unverified family checks.</p><p>Missing files remain in Missing history. Header problems and folder disagreements appear below. An unverified family is not a corrupt file; these checks do not approve an export.</p></>}
    {scan.error && <p>{scan.error}</p>}
    {scan.error && <button onClick={scan.retry} disabled={scan.pending}>Retry scan status</button>}
    {status?.status === "running" && <button onClick={scan.cancel} disabled={scan.pending || status.cancel_requested}>{status.cancel_requested ? "Stopping…" : "Stop background checks"}</button>}
    {status && ["partial", "cancelled"].includes(status.status) && status.remaining > 0 && <button onClick={scan.resume} disabled={scan.pending}>Continue unchecked files</button>}
    {error && <p role="alert">{error}</p>}
    {visibleIssues?.results?.map((item) => <article key={item.stable_id}><strong>{item.filename || item.stable_id}</strong>{item.source_status === "stale" && <small>Source changed; recheck required.</small>}{item.observation?.issues?.map((issue, index) => <p key={index}><span>{issue.severity === "not_checked" ? "Not verified" : "Potential issue"}</span> · {issue.message}</p>)}</article>)}
    {visibleIssues && <div className="studio-pagination"><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous observations</button><span>{visibleIssues.total ?? 0} files</span><button disabled={(page + 1) * 20 >= (visibleIssues.total ?? 0)} onClick={() => setPage(page + 1)}>More observations</button></div>}
  </div></details><LibraryLocation apiBase={apiBase} onChanged={freshness?.check} busy={scan.inventoryBusy || scan.status?.status === "running"} /></>;
}
