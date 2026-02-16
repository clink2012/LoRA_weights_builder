import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5001/api";
const PAGE_SIZE = 50;

const BASE_MODELS = [
  { code: "FLX", label: "Flux" },
  { code: "FLK", label: "Flux Krea" },
  { code: "W21", label: "WAN 2.1" },
  { code: "W22", label: "WAN 2.2" },
  { code: "PNY", label: "Pony" },
  { code: "SDX", label: "SDXL" },
  { code: "SD1", label: "SD 1.x" },
  { code: "ILL", label: "Illustrious" },
  { code: "ALL", label: "All Models" },
];

const CATEGORIES = [
  { code: "ALL", label: "All Categories" },
  { code: "PPL", label: "People" },
  { code: "STL", label: "Styles" },
  { code: "UTL", label: "Utils" },
  { code: "ACT", label: "Action" },
  { code: "BDY", label: "Body" },
  { code: "CHT", label: "Characters" },
  { code: "MCV", label: "Machines / Vehicles" },
  { code: "CLT", label: "Clothing" },
  { code: "ANM", label: "Animals" },
  { code: "BLD", label: "Buildings" },
  { code: "NAT", label: "Nature" },
];

function classNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

const COMMON_LAYOUT_OPTIONS = ["flux_fallback_16", "flux_unet_57", "unet_57"];

function formatLayoutLabel(layout) {
  if (!layout) return "Unknown layout";
  return layout.replaceAll("_", " ");
}

function getLoraTypeLabel(item) {
  return item?.lora_type?.trim() || "Unknown";
}

function getTypeBadge(item) {
  const raw = (getLoraTypeLabel(item) || "").toLowerCase();
  if (!raw || raw === "unknown") return "UNK";
  const hasUnet = raw.includes("unet");
  const hasText = raw.includes("text encoder") || raw.includes("text") || raw.includes("clip");
  if (hasUnet && hasText) return "U+T";
  if (hasUnet) return "UNET";
  if (hasText) return "TE";
  return "OTH";
}

function getLayoutBadge(layout) {
  if (!layout) return "LAY?";
  const normalized = layout.trim().toLowerCase();
  if (normalized === "flux_fallback_16") return "FBLK16";
  if (normalized === "unet_57" || normalized === "flux_unet_57") return "UNET57";
  const m = normalized.match(/^(flux_transformer|flux_double|flux_te|wan_unet|wan_[a-z0-9]+_unet)_(\d+)$/);
  if (!m) return "LAY?";
  const kind = m[1];
  const n = m[2];
  if (kind === "flux_transformer") return `TRF${n}`;
  if (kind === "flux_double") return `DBL${n}`;
  if (kind === "flux_te") return `TE${n}`;
  if (kind.startsWith("wan")) return `WUN${n}`;
  return `LAY${n}`;
}

function getExpectedBlocksFromLayout(layout) {
  if (!layout || typeof layout !== "string") return null;
  const normalized = layout.trim().toLowerCase();
  if (normalized === "flux_fallback_16") return 16;
  if (normalized === "unet_57" || normalized === "flux_unet_57") return 57;
  const match = normalized.match(/^(flux_transformer|flux_double|flux_te|wan_unet|wan_[a-z0-9]+_unet)_(\d+)$/);
  if (!match) return null;
  const count = Number.parseInt(match[2], 10);
  return Number.isFinite(count) ? count : null;
}

function getBlocksBadge(item) {
  const hasBlocks = Boolean(item?.has_block_weights);
  const expected = getExpectedBlocksFromLayout(item?.block_layout);
  if (hasBlocks) {
    const n = expected ?? (Number.isFinite(item?.block_count) ? item.block_count : null);
    return n ? `${n} BLKS` : "BLKS";
  }
  const baseCode = (item?.base_model_code || "").toUpperCase();
  if ((baseCode === "FLX" || baseCode === "FLK") && item?.block_layout === "flux_fallback_16") {
    return "16 FBLK";
  }
  return "NO BLKS";
}


function sortLoras(items, mode) {
  const data = [...items];
  if (mode === "name_asc" || mode === "name_desc") {
    data.sort((a, b) => {
      const an = (a.name || a.filename || "").toLowerCase();
      const bn = (b.name || b.filename || "").toLowerCase();
      if (an < bn) return -1;
      if (an > bn) return 1;
      return 0;
    });
    if (mode === "name_desc") data.reverse();
    return data;
  }
  if (mode === "date_new" || mode === "date_old") {
    data.sort((a, b) => (a.id ?? 0) - (b.id ?? 0));
    if (mode === "date_new") data.reverse();
    return data;
  }
  return data;
}

// --- Copy to clipboard helper ---
function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).catch(() => {
      fallbackCopy(text);
    });
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
}

// =====================================================================
// CopyButton component
// =====================================================================
function CopyButton({ text, label = "Copy" }) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef(null);

  function handleCopy(e) {
    e.stopPropagation();
    copyToClipboard(text);
    setCopied(true);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopied(false), 1500);
  }

  useEffect(() => () => clearTimeout(timerRef.current), []);

  return (
    <button
      className={classNames("lm-copy-btn", copied && "lm-copy-btn-done")}
      onClick={handleCopy}
      title={`Copy ${label}`}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

const BlockRow = memo(function BlockRow({ blockIndex, weight, isFallbackBlocks }) {
  const normalizedWeight = Number(weight) || 0;

  return (
    <div className={classNames("lm-block-row", isFallbackBlocks && "lm-block-row-fallback")}>
      <div className="lm-block-index">#{String(blockIndex ?? 0).padStart(2, "0")}</div>
      <div className="lm-block-bar-wrap">
        <div className="lm-block-bar-track">
          <div
            className="lm-block-bar-fill"
            style={{ width: `${Math.max(2, normalizedWeight * 100).toFixed(1)}%` }}
          />
        </div>
      </div>
      <div className="lm-block-value">{normalizedWeight.toFixed(3)}</div>
    </div>
  );
});

const BlockList = memo(function BlockList({ blocks, isFallbackBlocks }) {
  return (
    <div
      className={classNames("lm-blocks-list", isFallbackBlocks && "lm-blocks-list-fallback")}
      title="Block editing is not available yet. Editing arrives in Phase 6."
      aria-label="Block weight visualization, editing disabled until Phase 6"
    >
      {blocks.map((b) => (
        <BlockRow
          key={b.block_index}
          blockIndex={b.block_index}
          weight={b.weight}
          isFallbackBlocks={isFallbackBlocks}
        />
      ))}
    </div>
  );
});

const ProfileItem = memo(function ProfileItem({ profile, onLoad, onEdit, onDelete }) {
  return (
    <div className="lm-profile-item">
      <div className="lm-profile-item-info">
        <div className="lm-profile-item-name">{profile.profile_name}</div>
        <div className="lm-profile-item-meta">
          {profile.block_weights?.length ?? 0} blocks &middot; {profile.updated_at}
        </div>
      </div>
      <div className="lm-profile-item-actions">
        <button className="lm-action-btn lm-action-btn-sm" onClick={onLoad} title="Load this profile into view">
          Load
        </button>
        <button className="lm-action-btn lm-action-btn-sm" onClick={onEdit} title="Edit this profile">
          Edit
        </button>
        <button className="lm-action-btn lm-action-btn-sm lm-action-btn-danger" onClick={onDelete} title="Delete this profile">
          Delete
        </button>
      </div>
    </div>
  );
});

// =====================================================================
// Main App
// =====================================================================
function App() {
  const [baseModel, setBaseModel] = useState("FLX");
  const [category, setCategory] = useState("ALL");
  const [search, setSearch] = useState("");
  const [onlyBlocks, setOnlyBlocks] = useState(false);
  const [layoutFilter, setLayoutFilter] = useState("ALL_LAYOUTS");

  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const [selectedStableId, setSelectedStableId] = useState(null);
  const [selectedDetails, setSelectedDetails] = useState(null);
  const [blockData, setBlockData] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const [sortMode, setSortMode] = useState("name_asc");
  const [lastScanSummary, setLastScanSummary] = useState("");
  const [isRescanning, setIsRescanning] = useState(false);
  const [rescanProgress, setRescanProgress] = useState(null);

  // --- User profiles state ---
  const [profiles, setProfiles] = useState([]);
  const [profilesLoading, setProfilesLoading] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);

  // --- Profile editing state ---
  const [editingProfileId, setEditingProfileId] = useState(null);
  const [editingProfileName, setEditingProfileName] = useState("");

  // --- Copy weights button state ---
  const [copyWeightsStatus, setCopyWeightsStatus] = useState("idle"); // idle | copying | copied | failed

  const rescanPollRef = useRef(null);

  // Initial search on first load
  useEffect(() => {
    runSearch(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll index_status while rescanning
  useEffect(() => {
    if (!isRescanning) {
      clearInterval(rescanPollRef.current);
      rescanPollRef.current = null;
      return;
    }
    rescanPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/lora/index_status`);
        if (res.ok) {
          const data = await res.json();
          setRescanProgress(data);
          if (!data.indexing) {
            clearInterval(rescanPollRef.current);
          }
        }
      } catch {
        // ignore poll errors
      }
    }, 2000);
    return () => clearInterval(rescanPollRef.current);
  }, [isRescanning]);

  const currentBaseLabel =
    BASE_MODELS.find((b) => b.code === baseModel)?.label || "Unknown";
  const currentCategoryLabel =
    CATEGORIES.find((c) => c.code === category)?.label || "Unknown";

  const totalPages = Math.max(1, Math.ceil(totalResults / PAGE_SIZE));

  async function runSearch(page = 0) {
    try {
      setLoading(true);
      setErrorMsg("");

      const offset = page * PAGE_SIZE;
      const params = new URLSearchParams();

      if (baseModel && baseModel !== "ALL") params.set("base", baseModel);
      if (category && category !== "ALL") params.set("category", category);
      if (search.trim()) params.set("search", search.trim());
      if (onlyBlocks) params.set("has_blocks", "1");
      params.set("limit", String(PAGE_SIZE));
      params.set("offset", String(offset));

      const url = `${API_BASE}/lora/search?${params.toString()}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Search failed with status ${res.status}`);

      const data = await res.json();
      let list = Array.isArray(data.results) ? data.results : Array.isArray(data) ? data : [];

      const withBlockCount = list.map((item) => ({
        ...item,
        block_count: Number.isFinite(item?.block_count)
          ? item.block_count
          : item.has_block_weights
            ? getExpectedBlocksFromLayout(item.block_layout)
            : 0,
      }));

      const sorted = sortLoras(withBlockCount, sortMode);
      setResults(sorted);
      setTotalResults(data.total ?? sorted.length);

      // Clear selection when search results change to avoid showing details for LoRA not on current page
      // This includes: initial load, filter changes, and pagination
      setSelectedStableId(null);
      setSelectedDetails(null);
      setBlockData(null);
      setProfiles([]);

      setCurrentPage(page);
    } catch (err) {
      setErrorMsg(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function handlePageChange(newPage) {
    if (newPage < 0 || newPage >= totalPages) return;
    runSearch(newPage);
  }

  async function handleCardClick(item) {
    if (!item?.stable_id) return;
    const stableId = item.stable_id;

    try {
      setSelectedStableId(stableId);
      setSelectedDetails(null);
      setBlockData(null);
      setDetailsLoading(true);
      setProfiles([]);

      const [detailsRes, blocksRes] = await Promise.all([
        fetch(`${API_BASE}/lora/${stableId}`),
        fetch(`${API_BASE}/lora/${stableId}/blocks`),
      ]);

      if (!detailsRes.ok) throw new Error(`Details request failed (${detailsRes.status})`);

      const detailsJson = await detailsRes.json();
      setSelectedDetails(detailsJson);

      if (blocksRes.ok) {
        const blocksJson = await blocksRes.json();
        setBlockData(blocksJson);
      } else {
        setBlockData(null);
      }

      // Load user profiles
      loadProfiles(stableId);
    } catch (err) {
      setErrorMsg(err.message || "Failed to load details");
    } finally {
      setDetailsLoading(false);
    }
  }

  // --- Profile CRUD ---
  const loadProfiles = useCallback(async (stableId) => {
    try {
      setProfilesLoading(true);
      const res = await fetch(`${API_BASE}/lora/${stableId}/profiles`);
      if (res.ok) {
        const data = await res.json();
        setProfiles(data.profiles || []);
      }
    } catch {
      // silently fail
    } finally {
      setProfilesLoading(false);
    }
  }, []);

  async function handleSaveProfile() {
    if (!selectedStableId || !blockData?.blocks?.length) return;
    const name = newProfileName.trim();
    if (!name) return;

    try {
      setSavingProfile(true);
      const weights = blockData.blocks.map((b) => Number(b.weight) || 0);

      // Validate block weights (0.0 - 2.0 range)
      const invalidWeights = weights.filter((w) => w < 0.0 || w > 2.0);
      if (invalidWeights.length > 0) {
        throw new Error(`Invalid block weights detected. All weights must be between 0.0 and 2.0. Found ${invalidWeights.length} invalid value(s).`);
      }

      const res = await fetch(`${API_BASE}/lora/${selectedStableId}/profiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_name: name, block_weights: weights }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Save failed (${res.status})`);
      }
      setNewProfileName("");
      loadProfiles(selectedStableId);
    } catch (err) {
      setErrorMsg(err.message || "Failed to save profile");
    } finally {
      setSavingProfile(false);
    }
  }

  function handleEditProfile(profile) {
    if (!profile?.id || !profile?.profile_name) return;
    setEditingProfileId(profile.id);
    setEditingProfileName(profile.profile_name);

    // Load the profile weights into the block data for editing
    if (profile.block_weights?.length) {
      setBlockData((prev) => {
        if (!prev) return prev;
        const newBlocks = profile.block_weights.map((w, i) => ({
          block_index: i,
          weight: w,
          raw_strength: prev.blocks?.[i]?.raw_strength ?? null,
        }));
        return { ...prev, blocks: newBlocks, fallback: false, fallback_reason: null };
      });
    }
  }

  async function handleUpdateProfile() {
    if (!selectedStableId || !editingProfileId || !blockData?.blocks?.length) return;
    const name = editingProfileName.trim();
    if (!name) return;

    try {
      setSavingProfile(true);
      const weights = blockData.blocks.map((b) => Number(b.weight) || 0);

      // Validate block weights (0.0 - 2.0 range)
      const invalidWeights = weights.filter((w) => w < 0.0 || w > 2.0);
      if (invalidWeights.length > 0) {
        throw new Error(`Invalid block weights detected. All weights must be between 0.0 and 2.0. Found ${invalidWeights.length} invalid value(s).`);
      }

      const res = await fetch(`${API_BASE}/lora/${selectedStableId}/profiles/${editingProfileId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_name: name, block_weights: weights }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Update failed (${res.status})`);
      }
      setEditingProfileId(null);
      setEditingProfileName("");
      loadProfiles(selectedStableId);
    } catch (err) {
      setErrorMsg(err.message || "Failed to update profile");
    } finally {
      setSavingProfile(false);
    }
  }

  function handleCancelEdit() {
    setEditingProfileId(null);
    setEditingProfileName("");
    // Optionally reload original weights here if needed
  }

  async function handleDeleteProfile(profileId) {
    if (!selectedStableId) return;
    if (!window.confirm("Delete this saved profile?")) return;
    try {
      const res = await fetch(`${API_BASE}/lora/${selectedStableId}/profiles/${profileId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);

      // If we were editing this profile, cancel the edit
      if (editingProfileId === profileId) {
        setEditingProfileId(null);
        setEditingProfileName("");
      }

      loadProfiles(selectedStableId);
    } catch (err) {
      setErrorMsg(err.message || "Failed to delete profile");
    }
  }

  async function handleLoadProfile(profile) {
    if (!profile?.block_weights?.length) return;

    // Validate block weights before loading
    const invalidWeights = profile.block_weights.filter((w) => w < 0.0 || w > 2.0);
    if (invalidWeights.length > 0) {
      setErrorMsg(`Cannot load profile: contains ${invalidWeights.length} invalid weight(s). All weights must be between 0.0 and 2.0.`);
      return;
    }

    setBlockData((prev) => {
      if (!prev) return prev;
      const newBlocks = profile.block_weights.map((w, i) => ({
        block_index: i,
        weight: w,
        raw_strength: prev.blocks?.[i]?.raw_strength ?? null,
      }));
      return { ...prev, blocks: newBlocks, fallback: false, fallback_reason: null };
    });
  }

  function handleSearchSubmit(e) {
    e.preventDefault();
    runSearch(0);
  }

  async function handleFullRescan() {
    const confirmed = window.confirm(
      "Full rescan & reindex ALL LoRAs?\n\n" +
      "This can take a while if you have a lot of files.\n" +
      "The list will refresh automatically when it finishes."
    );
    if (!confirmed) return;

    try {
      setIsRescanning(true);
      setRescanProgress(null);
      setErrorMsg("");

      const res = await fetch(`${API_BASE}/lora/reindex_all`, { method: "POST" });
      if (!res.ok) throw new Error(`Reindex failed with status ${res.status}`);

      const info = await res.json();
      const s = info.summary || {};
      const summaryText = `Indexed ${s.total ?? 0} LoRAs \u00b7 With blocks: ${s.with_blocks ?? 0} \u00b7 No blocks: ${s.no_blocks ?? 0} \u00b7 ${info.duration_sec ?? 0}s`;
      setLastScanSummary(summaryText);
      await runSearch(0);
    } catch (err) {
      setLastScanSummary("Rescan failed \u2013 check backend logs.");
      window.alert(err.message || "Reindex failed \u2013 see backend console.");
    } finally {
      setIsRescanning(false);
      setRescanProgress(null);
    }
  }

  async function handleExportCsv() {
    if (!selectedStableId) return;
    window.open(`${API_BASE}/lora/${selectedStableId}/export`, "_blank");
  }

  async function handleCopyWeights() {
    if (!blockData?.blocks?.length) return;

    setCopyWeightsStatus("copying");
    const weightsStr = blockData.blocks.map((b) => Number(b.weight).toFixed(3)).join(",");

    try {
      await copyToClipboard(weightsStr);
      setCopyWeightsStatus("copied");
      setTimeout(() => setCopyWeightsStatus("idle"), 2000);
    } catch {
      setCopyWeightsStatus("failed");
      setTimeout(() => setCopyWeightsStatus("idle"), 2000);
    }
  }

  const sortedResults = sortLoras(results, sortMode);
  const layoutOptions = useMemo(() => {
    const fromResults = results
      .map((item) => item?.block_layout)
      .filter((layout) => typeof layout === "string" && layout.trim().length > 0)
      .map((layout) => layout.trim().toLowerCase());
    return [...new Set([...COMMON_LAYOUT_OPTIONS, ...fromResults])];
  }, [results]);

  useEffect(() => {
    if (layoutFilter === "ALL_LAYOUTS") return;
    if (!loading && results.length > 0 && !layoutOptions.includes(layoutFilter)) {
      setLayoutFilter("ALL_LAYOUTS");
    }
  }, [loading, results, layoutFilter, layoutOptions]);

  const filteredResults =
    layoutFilter === "ALL_LAYOUTS"
      ? sortedResults
      : sortedResults.filter(
        (item) => (item?.block_layout || "").toLowerCase() === layoutFilter
      );

  const apiBaseDisplay = API_BASE.replace("/api", "");
  const hasAnyBlocks = Array.isArray(blockData?.blocks) && blockData.blocks.length > 0;
  const isFallbackBlocks = Boolean(blockData?.fallback);
  const fallbackReason = blockData?.fallback_reason || "Fallback profile generated for this layout.";
  const blockRows = useMemo(() => (Array.isArray(blockData?.blocks) ? blockData.blocks : []), [blockData]);

  return (
    <div className="lm-app">
      {/* Left Sidebar */}
      <aside className="lm-sidebar">
        <div className="lm-brand">
          <div className="lm-logo-circle">
            <span className="lm-logo-text">LM</span>
          </div>
          <div className="lm-brand-text">
            <div className="lm-brand-title">LORA</div>
            <div className="lm-brand-subtitle">MASTER</div>
          </div>
        </div>

        <form className="lm-filters" onSubmit={handleSearchSubmit}>
          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="base-model">Base model</label>
            <select id="base-model" className="lm-select" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
              {BASE_MODELS.map((m) => <option key={m.code} value={m.code}>{m.label}</option>)}
            </select>
          </div>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="category">Category</label>
            <select id="category" className="lm-select" value={category} onChange={(e) => setCategory(e.target.value)}>
              {CATEGORIES.map((c) => <option key={c.code} value={c.code}>{c.label}</option>)}
            </select>
          </div>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="search">Search</label>
            <input id="search" className="lm-input" type="text" placeholder="Filename contains..." value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>

          <label className="lm-checkbox-row">
            <input type="checkbox" checked={onlyBlocks} onChange={(e) => setOnlyBlocks(e.target.checked)} />
            <span>Only LoRAs with block weights</span>
          </label>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="layout-filter">Block layout</label>
            <select id="layout-filter" className="lm-select" value={layoutFilter} onChange={(e) => setLayoutFilter(e.target.value)}>
              <option value="ALL_LAYOUTS">All layouts</option>
              {layoutOptions.map((layout) => <option key={layout} value={layout}>{formatLayoutLabel(layout)}</option>)}
            </select>
          </div>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="sort-mode">Sort</label>
            <select
              id="sort-mode"
              className="lm-select"
              value={sortMode}
              onChange={(e) => {
                const mode = e.target.value;
                setSortMode(mode);
                setResults((prev) => sortLoras(prev, mode));
              }}
            >
              <option value="name_asc">Name &middot; A &rarr; Z</option>
              <option value="name_desc">Name &middot; Z &rarr; A</option>
              <option value="date_new">Date added &middot; Newest</option>
              <option value="date_old">Date added &middot; Oldest</option>
            </select>
          </div>

          <div className="lm-filter-actions">
            <button type="submit" className="lm-button" disabled={loading} title="Run search with current filters">
              {loading ? "Searching..." : "Run search"}
            </button>
            <button type="button" className="lm-button lm-button-secondary" onClick={handleFullRescan} disabled={isRescanning} title="Full rescan & reindex all LoRAs">
              {isRescanning ? "Rescanning..." : "Rescan & reindex"}
            </button>
          </div>
        </form>

        {/* Rescan progress indicator */}
        {isRescanning && (
          <div className="lm-rescan-progress">
            <div className="lm-rescan-bar">
              <div className="lm-rescan-bar-fill" />
            </div>
            <div className="lm-rescan-text">
              {rescanProgress?.indexing ? "Indexing in progress..." : "Starting rescan..."}
            </div>
          </div>
        )}

        <div className="lm-sidebar-footer">
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-badge">
              {totalResults} total &middot; page {currentPage + 1}/{totalPages}
            </span>
          </div>
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-backend">Backend: {apiBaseDisplay}</span>
          </div>
          {lastScanSummary && <div className="lm-last-scan">{lastScanSummary}</div>}
        </div>
      </aside>

      {/* Main content */}
      <main className="lm-main">
        <header className="lm-main-header">
          <div>
            <div className="lm-main-pill-row">
              <span className="lm-main-pill lm-main-pill-active">Dashboard</span>
              <span className="lm-main-pill">Patterns</span>
              <span className="lm-main-pill">Compare</span>
              <span className="lm-main-pill">Delta Lab</span>
            </div>
            <div className="lm-main-subtitle">
              {currentBaseLabel} / {currentCategoryLabel} / {onlyBlocks ? "Block-weighted only" : "All LoRAs"}
            </div>
          </div>
        </header>

        <section className="lm-layout">
          {/* Search results */}
          <section className="lm-results">
            <div className="lm-results-header">
              <div className="lm-results-title">LoRA catalog</div>
              <div className="lm-results-count">{filteredResults.length} in view &middot; {totalResults} total</div>
            </div>

            {errorMsg && (
              <div className="lm-error-banner"><span>{errorMsg}</span></div>
            )}

            {!loading && filteredResults.length === 0 && !errorMsg && (
              <div className="lm-empty-state">No LoRAs match the current filters.</div>
            )}

            {filteredResults.length > 0 && (
              <div className="lm-results-grid">
                {filteredResults.map((item) => {
                  const hasBlocks = Boolean(item.has_block_weights);
                  const isSelected = item.stable_id && item.stable_id === selectedStableId;

                  return (
                    <article
                      key={item.id}
                      className={classNames("lm-card", isSelected && "lm-card-selected")}
                      onClick={() => handleCardClick(item)}
                    >
                      <div className="lm-card-header">
                        <div className="lm-card-id">{item.stable_id || "UNASSIGNED"}</div>
                        <div className={classNames("lm-card-badge", hasBlocks ? "lm-badge-blocks" : "lm-badge-noblocks")}>
                          {getBlocksBadge(item)}
                        </div>
                      </div>
                      <div className="lm-card-filename" title={item.filename || ""}>{item.filename}</div>
                      <div className="lm-card-path">{(item.file_path || "").replace(/\\/g, "/")}</div>
                      <div className="lm-card-footer">
                        <span className="lm-chip">{item.base_model_code}</span>
                        <span className="lm-chip lm-chip-soft">{item.category_code}</span>
                        <span className="lm-chip lm-chip-soft" title={item.block_layout || ""}>{getLayoutBadge(item.block_layout)}</span>
                        <span className="lm-chip lm-chip-type" title={getLoraTypeLabel(item)}>{getTypeBadge(item)}</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}

            {/* Pagination controls */}
            {totalPages > 1 && (
              <div className="lm-pagination">
                <button
                  className="lm-page-btn"
                  disabled={currentPage === 0}
                  onClick={() => handlePageChange(0)}
                  title="First page"
                >
                  &laquo;
                </button>
                <button
                  className="lm-page-btn"
                  disabled={currentPage === 0}
                  onClick={() => handlePageChange(currentPage - 1)}
                  title="Previous page"
                >
                  &lsaquo;
                </button>
                <span className="lm-page-info">
                  {currentPage + 1} / {totalPages}
                </span>
                <button
                  className="lm-page-btn"
                  disabled={currentPage >= totalPages - 1}
                  onClick={() => handlePageChange(currentPage + 1)}
                  title="Next page"
                >
                  &rsaquo;
                </button>
                <button
                  className="lm-page-btn"
                  disabled={currentPage >= totalPages - 1}
                  onClick={() => handlePageChange(totalPages - 1)}
                  title="Last page"
                >
                  &raquo;
                </button>
              </div>
            )}
          </section>

          {/* Details panel */}
          <section className="lm-details">
            <div className="lm-details-card">
              <header className="lm-details-header">
                <div className="lm-details-title-block">
                  <div className="lm-details-filename">
                    {selectedDetails?.filename || "Select a LoRA"}
                  </div>
                </div>
                {selectedDetails?.stable_id && (
                  <div className="lm-details-id-stack">
                    <div className="lm-details-stable-id-row">
                      <div className="lm-details-stable-id">{selectedDetails.stable_id}</div>
                    </div>
                    <div className="lm-details-lora-type-pill">{getLoraTypeLabel(selectedDetails)}</div>
                  </div>
                )}
              </header>

              {detailsLoading && (
                <div className="lm-details-loading">Loading details...</div>
              )}

              {!detailsLoading && selectedDetails && (
                <>
                  <dl className="lm-details-grid">
                    <dt>Path</dt>
                    <dd className="lm-dd-path">
                      <span title={selectedDetails.file_path}>{selectedDetails.file_path}</span>
                    </dd>
                  </dl>

                  <div className="lm-details-meta">
                    <span>Created: {selectedDetails.created_at || "not recorded"}</span>
                    <span>Updated: {selectedDetails.updated_at || "not recorded"}</span>
                  </div>

                  {/* Action buttons row */}
                  {hasAnyBlocks && !isFallbackBlocks && (
                    <div className="lm-details-actions">
                      <button className="lm-action-btn" onClick={handleExportCsv} title="Export block weights as CSV">
                        Export CSV
                      </button>
                      <button
                        className="lm-action-btn"
                        onClick={handleCopyWeights}
                        disabled={copyWeightsStatus === "copying"}
                        title="Copy block weights as comma-separated values"
                      >
                        {copyWeightsStatus === "copied" ? "Copied!" :
                          copyWeightsStatus === "failed" ? "Copy failed" :
                            copyWeightsStatus === "copying" ? "Copying..." : "Copy Weights"}
                      </button>
                    </div>
                  )}
                </>
              )}

              {/* Block weights */}
              <div className="lm-blocks-panel">
                <div className="lm-blocks-header">
                  <div className="lm-blocks-title-row">
                    <div className="lm-blocks-title">Block weights</div>
                    <span className="lm-blocks-readonly" title="Editing coming in Phase 6">
                      Read-only · Phase 6
                    </span>
                    {isFallbackBlocks && (
                      <span className="lm-fallback-badge" title={fallbackReason}>FALLBACK</span>
                    )}
                  </div>
                  <div className="lm-blocks-count">
                    {hasAnyBlocks ? `${blockData.blocks.length} blocks` : "No block weights"}
                  </div>
                </div>

                {isFallbackBlocks && selectedDetails && (
                  <div className="lm-fallback-note" title={fallbackReason}>{fallbackReason}</div>
                )}

                {hasAnyBlocks && <BlockList blocks={blockRows} isFallbackBlocks={isFallbackBlocks} />}

                {!detailsLoading && selectedDetails && !hasAnyBlocks && (
                  <div className="lm-blocks-empty">
                    {isFallbackBlocks
                      ? "Showing fallback block profile (not extracted weights)."
                      : "This LoRA has no recorded block weights in the database."}
                  </div>
                )}

                {!selectedDetails && !detailsLoading && (
                  <div className="lm-blocks-empty">Select a LoRA card to view its block profile.</div>
                )}
              </div>

              {/* User profiles section */}
              {selectedDetails && (
                <div className="lm-profiles-panel">
                  <div className="lm-profiles-header">
                    <div className="lm-profiles-title">Saved profiles</div>
                    <div className="lm-profiles-count">
                      {profilesLoading ? "Loading..." : `${profiles.length} profile${profiles.length === 1 ? "" : "s"}`}
                    </div>
                  </div>

                  {/* Save/Edit profile row */}
                  {hasAnyBlocks && (
                    <div className="lm-profile-create-block">
                      <div className="lm-profile-create-label">Create profile</div>
                      <div className="lm-profile-save-row">
                        <input
                          className="lm-input lm-profile-name-input"
                          type="text"
                          placeholder={editingProfileId ? "Edit profile name..." : "Profile name..."}
                          value={editingProfileId ? editingProfileName : newProfileName}
                          onChange={(e) => editingProfileId ? setEditingProfileName(e.target.value) : setNewProfileName(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") editingProfileId ? handleUpdateProfile() : handleSaveProfile(); }}
                        />
                        {editingProfileId ? (
                          <>
                            <button
                              className="lm-action-btn"
                              onClick={handleUpdateProfile}
                              disabled={savingProfile || !editingProfileName.trim()}
                            >
                              {savingProfile ? "Updating..." : "Update"}
                            </button>
                            <button
                              className="lm-action-btn lm-action-btn-sm"
                              onClick={handleCancelEdit}
                              disabled={savingProfile}
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <button
                            className="lm-action-btn"
                            onClick={handleSaveProfile}
                            disabled={savingProfile || !newProfileName.trim()}
                          >
                            {savingProfile ? "Saving..." : "Save"}
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Profile list */}
                  {profiles.length > 0 && (
                    <div className="lm-profile-list">
                      {profiles.map((p) => (
                        <ProfileItem
                          key={p.id}
                          profile={p}
                          onLoad={() => handleLoadProfile(p)}
                          onEdit={() => handleEditProfile(p)}
                          onDelete={() => handleDeleteProfile(p.id)}
                        />
                      ))}
                    </div>
                  )}

                  {!profilesLoading && profiles.length === 0 && (
                    <div className="lm-profiles-empty">No saved profiles yet.</div>
                  )}
                </div>
              )}
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
