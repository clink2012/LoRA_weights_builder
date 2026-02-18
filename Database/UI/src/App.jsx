import { Component, memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";
const PAGE_SIZE = 50;
const DASHBOARD_TAB = "dashboard";
const COMBINE_TAB = "combine";

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

function bannerString(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (value instanceof Error) return value.message || String(value);
  if (Array.isArray(value)) return value.map(bannerString).filter(Boolean).join("; ");
  try {
    const json = JSON.stringify(value);
    return json && json !== "{}" ? json : String(value);
  } catch {
    return String(value);
  }
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

function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text).catch(() => {
      fallbackCopy(text);
    });
  }
  fallbackCopy(text);
  return Promise.resolve();
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

function formatDateOnly(value) {
  if (!value) return "";
  const s = String(value);
  if (s.length >= 10 && /^\d{4}-\d{2}-\d{2}/.test(s)) return s.slice(0, 10);
  return s;
}

function clampBlockWeight(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function parseWeightInput(value) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) return null;
  return clampBlockWeight(parsed);
}

function toOneDecimalWeight(value) {
  return Number(clampBlockWeight(value).toFixed(1));
}

class BlockPanelErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error) {
    console.error("Block panel render error:", error);
  }

  render() {
    if (this.state.hasError) {
      return <div className="lm-blocks-empty">Block panel failed to render. Please reselect the LoRA.</div>;
    }
    return this.props.children;
  }
}

const BlockRow = memo(function BlockRow({
  block,
  compactMode,
  showSlider,
  isFallbackBlocks,
  isDirty,
  onWeightChange,
  onReset,
}) {
  const safeWeight = clampBlockWeight(Number(block.weight) || 0);
  const barTrackRef = useRef(null);
  const draggingRef = useRef(false);

  const updateFromPointerPosition = useCallback(
    (clientX) => {
      const trackEl = barTrackRef.current;
      if (!trackEl) return;
      const rect = trackEl.getBoundingClientRect();
      if (!rect.width) return;
      const ratio = clampBlockWeight((clientX - rect.left) / rect.width);
      onWeightChange(block.block_index, ratio);
    },
    [block.block_index, onWeightChange]
  );

  const handleTrackPointerDown = useCallback(
    (e) => {
      draggingRef.current = true;
      e.currentTarget.setPointerCapture?.(e.pointerId);
      updateFromPointerPosition(e.clientX);
    },
    [updateFromPointerPosition]
  );

  const handleTrackPointerMove = useCallback(
    (e) => {
      if (!draggingRef.current) return;
      updateFromPointerPosition(e.clientX);
    },
    [updateFromPointerPosition]
  );

  const handleTrackPointerUp = useCallback((e) => {
    draggingRef.current = false;
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  }, []);

  return (
    <div
      className={classNames(
        "lm-block-row",
        compactMode && "lm-block-row-compact",
        isFallbackBlocks && "lm-block-row-fallback",
        isDirty && "lm-block-row-dirty"
      )}
      key={block.block_index}
    >
      <div className="lm-block-index">#{String(block.block_index ?? 0).padStart(2, "0")}</div>
      <div className="lm-block-main">
        <div className="lm-block-bar-wrap">
          <div
            ref={barTrackRef}
            className="lm-block-bar-track"
            data-testid={`block-bar-track-${block.block_index}`}
            onPointerDown={handleTrackPointerDown}
            onPointerMove={handleTrackPointerMove}
            onPointerUp={handleTrackPointerUp}
            onPointerCancel={handleTrackPointerUp}
          >
            <div className="lm-block-bar-fill" style={{ width: `${Math.max(2, safeWeight * 100).toFixed(1)}%` }} />
          </div>
        </div>
        {showSlider && (
          <input
            className="lm-block-slider"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={safeWeight}
            onChange={(e) => onWeightChange(block.block_index, e.target.value)}
            aria-label={`Block ${block.block_index} slider`}
          />
        )}
      </div>
      <input
        className="lm-block-edit-input"
        type="number"
        min="0"
        max="1"
        step="0.1"
        value={safeWeight.toFixed(1)}
        onChange={(e) => onWeightChange(block.block_index, e.target.value)}
        aria-label={`Block ${block.block_index} weight`}
      />
      <button
        className="lm-action-btn lm-action-btn-sm"
        type="button"
        onClick={() => onReset(block.block_index)}
        disabled={!isDirty}
        title={isDirty ? "Reset this block" : "No edits for this block"}
      >
        Reset
      </button>
    </div>
  );
});

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
    <button className={classNames("lm-copy-btn", copied && "lm-copy-btn-done")} onClick={handleCopy} title={`Copy ${label}`}>
      {copied ? "Copied" : label}
    </button>
  );
}

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
  const [warningMsg, setWarningMsg] = useState("");

  // Dashboard selection (single item)
  const [selectedStableId, setSelectedStableId] = useState(null);
  const [selectedDetails, setSelectedDetails] = useState(null);
  const [blockData, setBlockData] = useState(null);
  const [originalBlockWeights, setOriginalBlockWeights] = useState([]);
  const [extractedBlockWeights, setExtractedBlockWeights] = useState([]);
  const [activeWeightsView, setActiveWeightsView] = useState({ type: "default", label: "Default" });
  const [detailsLoading, setDetailsLoading] = useState(false);

  const [sortMode, setSortMode] = useState("name_asc");
  const [lastScanSummary, setLastScanSummary] = useState("");
  const [isRescanning, setIsRescanning] = useState(false);
  const [rescanProgress, setRescanProgress] = useState(null);

  // Profiles
  const [profiles, setProfiles] = useState([]);
  const [profilesLoading, setProfilesLoading] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [editingProfileId, setEditingProfileId] = useState(null);
  const [editingProfileName, setEditingProfileName] = useState("");

  // Copy
  const [copyWeightsStatus, setCopyWeightsStatus] = useState("idle");
  const [compactMode] = useState(true);

  // Tabs
  const [activeTab, setActiveTab] = useState(DASHBOARD_TAB);

  // Combine workbench state
  const [combineSearch, setCombineSearch] = useState("");
  const [combineSelectedIds, setCombineSelectedIds] = useState([]);
  const [combineShowAll, setCombineShowAll] = useState(false);
  const [combineLoading, setCombineLoading] = useState(false);
  const [combineError, setCombineError] = useState("");
  const [combineResult, setCombineResult] = useState(null);

  const rescanPollRef = useRef(null);

  useEffect(() => {
    runSearch(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        // ignore
      }
    }, 2000);
    return () => clearInterval(rescanPollRef.current);
  }, [isRescanning]);

  const currentBaseLabel = BASE_MODELS.find((b) => b.code === baseModel)?.label || "Unknown";
  const currentCategoryLabel = CATEGORIES.find((c) => c.code === category)?.label || "Unknown";

  const totalPages = Math.max(1, Math.ceil(totalResults / PAGE_SIZE));

  const hasAnyBlocks = Array.isArray(blockData?.blocks) && blockData.blocks.length > 0;
  const isFallbackBlocks = Boolean(blockData?.fallback);
  const fallbackReason = blockData?.fallback_reason || "Neutral fallback profile";
  const effectiveShowSliders = false;

  const dirtyByIndex = useMemo(() => {
    if (!hasAnyBlocks) return {};
    const map = {};
    blockData.blocks.forEach((b, idx) => {
      const current = clampBlockWeight(Number(b.weight) || 0);
      const original = clampBlockWeight(Number(originalBlockWeights[idx]) || 0);
      map[b.block_index] = Math.abs(current - original) > 0.0001;
    });
    return map;
  }, [blockData, hasAnyBlocks, originalBlockWeights]);

  const isDirty = useMemo(
    () =>
      hasAnyBlocks &&
      blockData.blocks.some(
        (b, idx) => Math.abs(clampBlockWeight(Number(b.weight) || 0) - clampBlockWeight(Number(originalBlockWeights[idx]) || 0)) > 0.0001
      ),
    [blockData, hasAnyBlocks, originalBlockWeights]
  );

  const blockStats = useMemo(() => {
    if (!hasAnyBlocks) return null;
    const weights = blockData.blocks.map((b) => clampBlockWeight(Number(b.weight) || 0));
    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const mean = weights.reduce((sum, w) => sum + w, 0) / weights.length;
    const variance = weights.reduce((sum, w) => sum + (w - mean) ** 2, 0) / weights.length;
    return { min, max, mean, variance };
  }, [blockData, hasAnyBlocks]);

  useEffect(() => {
    const onBeforeUnload = (event) => {
      if (!isDirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [isDirty]);

  async function runSearch(page = 0) {
    if (isDirty) {
      const confirmed = window.confirm("You have unsaved block edits.\n\nRunning a new search will discard them. Continue?");
      if (!confirmed) return;
    }

    try {
      setLoading(true);
      setErrorMsg("");
      setWarningMsg("");

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

      // Clear dashboard details selection on new searches
      setSelectedStableId(null);
      setSelectedDetails(null);
      setBlockData(null);
      setOriginalBlockWeights([]);
      setExtractedBlockWeights([]);
      setActiveWeightsView({ type: "default", label: "Default" });
      setProfiles([]);

      // Combine results should be deterministic: clear computed output when catalog changes
      setCombineError("");
      setCombineResult(null);

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

  const loadProfiles = useCallback(async (stableId) => {
    try {
      setProfilesLoading(true);
      const res = await fetch(`${API_BASE}/lora/${stableId}/profiles`);
      if (res.ok) {
        const data = await res.json();
        setProfiles(data.profiles || []);
      }
    } catch {
      // ignore
    } finally {
      setProfilesLoading(false);
    }
  }, []);

  async function handleCardClick(item) {
    if (!item?.stable_id) return;
    const stableId = item.stable_id;

    if (isDirty) {
      const confirmed = window.confirm("You have unsaved block edits. Switch LoRA and discard changes?");
      if (!confirmed) return;
    }

    try {
      setSelectedStableId(stableId);
      setSelectedDetails(null);
      setBlockData(null);
      setWarningMsg("");
      setOriginalBlockWeights([]);
      setExtractedBlockWeights([]);
      setActiveWeightsView({ type: "default", label: "Default" });
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
        const sanitizedBlocks = Array.isArray(blocksJson?.blocks)
          ? blocksJson.blocks.map((b) => ({
              ...b,
              weight: clampBlockWeight(Number(b.weight) || 0),
            }))
          : [];
        const baseline = sanitizedBlocks.map((b) => clampBlockWeight(Number(b.weight) || 0));
        setBlockData({ ...blocksJson, blocks: sanitizedBlocks });
        setOriginalBlockWeights(baseline);
        setExtractedBlockWeights(baseline);
        setActiveWeightsView({ type: "default", label: "Default" });
      } else {
        setBlockData(null);
        setOriginalBlockWeights([]);
        setExtractedBlockWeights([]);
        setActiveWeightsView({ type: "default", label: "Default" });
      }

      loadProfiles(stableId);
    } catch (err) {
      setErrorMsg(err.message || "Failed to load details");
    } finally {
      setDetailsLoading(false);
    }
  }

  async function handleSaveProfile() {
    if (!selectedStableId || !blockData?.blocks?.length) return;
    const name = newProfileName.trim();
    if (!name) return;

    try {
      setSavingProfile(true);
      const weights = blockData.blocks.map((b) => toOneDecimalWeight(Number(b.weight) || 0));

      const res = await fetch(`${API_BASE}/lora/${selectedStableId}/profiles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_name: name, block_weights: weights }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Save failed (${res.status})`);
      }

      const normalizedWeights = weights.map((w) => clampBlockWeight(Number(w) || 0));
      setBlockData((prev) => {
        if (!prev?.blocks?.length) return prev;
        const nextBlocks = prev.blocks.map((b, idx) => ({
          ...b,
          weight: normalizedWeights[idx] ?? 0,
        }));
        return { ...prev, blocks: nextBlocks, fallback: false, fallback_reason: null };
      });

      setNewProfileName("");
      setOriginalBlockWeights(normalizedWeights);
      setActiveWeightsView({ type: "profile", label: name });
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
    setActiveWeightsView({ type: "profile", label: profile.profile_name || "Saved profile" });

    if (profile.block_weights?.length) {
      const clampedCount = profile.block_weights.reduce((count, w) => {
        const numeric = Number(w);
        if (!Number.isFinite(numeric)) return count + 1;
        return numeric < 0 || numeric > 1 ? count + 1 : count;
      }, 0);
      if (clampedCount > 0) {
        setWarningMsg(`Loaded profile "${profile.profile_name || "Unnamed"}" with ${clampedCount} legacy value(s) clamped to 0.0–1.0.`);
      } else {
        setWarningMsg("");
      }

      setBlockData((prev) => {
        if (!prev) return prev;
        const newBlocks = profile.block_weights.map((w, i) => ({
          block_index: i,
          weight: clampBlockWeight(Number(w) || 0),
          raw_strength: prev.blocks?.[i]?.raw_strength ?? null,
        }));
        return { ...prev, blocks: newBlocks, fallback: false, fallback_reason: null };
      });
      setOriginalBlockWeights(profile.block_weights.map((w) => clampBlockWeight(Number(w) || 0)));
    }
  }

  async function handleUpdateProfile() {
    if (!selectedStableId || !editingProfileId || !blockData?.blocks?.length) return;
    const name = editingProfileName.trim();
    if (!name) return;

    try {
      setSavingProfile(true);
      const weights = blockData.blocks.map((b) => toOneDecimalWeight(Number(b.weight) || 0));

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
      setOriginalBlockWeights(weights.map((w) => clampBlockWeight(Number(w) || 0)));
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
  }

  async function handleDeleteProfile(profileId) {
    if (!selectedStableId) return;
    if (!window.confirm("Delete this saved profile?")) return;
    try {
      const res = await fetch(`${API_BASE}/lora/${selectedStableId}/profiles/${profileId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);

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

    const clampedCount = profile.block_weights.reduce((count, w) => {
      const numeric = Number(w);
      if (!Number.isFinite(numeric)) return count + 1;
      return numeric < 0 || numeric > 1 ? count + 1 : count;
    }, 0);
    if (clampedCount > 0) {
      setWarningMsg(`Loaded profile "${profile.profile_name || "Unnamed"}" with ${clampedCount} legacy value(s) clamped to 0.0–1.0.`);
    } else {
      setWarningMsg("");
    }

    setActiveWeightsView({ type: "profile", label: profile.profile_name || "Saved profile" });
    setBlockData((prev) => {
      if (!prev) return prev;
      const newBlocks = profile.block_weights.map((w, i) => ({
        block_index: i,
        weight: clampBlockWeight(Number(w) || 0),
        raw_strength: prev.blocks?.[i]?.raw_strength ?? null,
      }));
      return { ...prev, blocks: newBlocks, fallback: false, fallback_reason: null };
    });

    setOriginalBlockWeights(profile.block_weights.map((w) => clampBlockWeight(Number(w) || 0)));
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
      const summaryText = `Indexed ${s.total ?? 0} LoRAs · With blocks: ${s.with_blocks ?? 0} · No blocks: ${s.no_blocks ?? 0} · ${info.duration_sec ?? 0}s`;
      setLastScanSummary(summaryText);
      await runSearch(0);
    } catch (err) {
      setLastScanSummary("Rescan failed – check backend logs.");
      window.alert(err.message || "Reindex failed – see backend console.");
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
    const weightsStr = blockData.blocks
      .map((b) => Math.max(0, Math.min(1, Number(b.weight) || 0)).toFixed(1))
      .join(",");

    try {
      await copyToClipboard(weightsStr);
      setCopyWeightsStatus("copied");
      setTimeout(() => setCopyWeightsStatus("idle"), 2000);
    } catch {
      setCopyWeightsStatus("failed");
      setTimeout(() => setCopyWeightsStatus("idle"), 2000);
    }
  }

  const handleBlockWeightChange = useCallback((blockIndex, rawValue) => {
    setBlockData((prev) => {
      if (!prev?.blocks?.length) return prev;
      const parsed = parseWeightInput(rawValue);
      if (parsed === null) return prev;
      const nextBlocks = prev.blocks.map((b) => {
        if (b.block_index !== blockIndex) return b;
        return { ...b, weight: parsed };
      });
      return { ...prev, blocks: nextBlocks };
    });
  }, []);

  const handleResetSingleBlock = useCallback(
    (blockIndex) => {
      setBlockData((prev) => {
        if (!prev?.blocks?.length) return prev;
        const nextBlocks = prev.blocks.map((b, idx) => {
          if (b.block_index !== blockIndex) return b;
          return { ...b, weight: clampBlockWeight(Number(originalBlockWeights[idx]) || 0) };
        });
        return { ...prev, blocks: nextBlocks };
      });
    },
    [originalBlockWeights]
  );

  const handleResetAllBlocks = useCallback(() => {
    setBlockData((prev) => {
      if (!prev?.blocks?.length) return prev;
      const nextBlocks = prev.blocks.map((b, idx) => ({
        ...b,
        weight: clampBlockWeight(Number(originalBlockWeights[idx]) || 0),
      }));
      return { ...prev, blocks: nextBlocks };
    });
    setActiveWeightsView({ type: "default", label: "Default" });
  }, [originalBlockWeights]);

  const handleUseDefaultWeights = useCallback(() => {
    if (!blockData?.blocks?.length || !extractedBlockWeights.length) return;
    const currentDirty = blockData.blocks.some(
      (b, idx) =>
        Math.abs(clampBlockWeight(Number(b.weight) || 0) - clampBlockWeight(Number(originalBlockWeights[idx]) || 0)) > 0.0001
    );
    if (currentDirty) {
      const confirmed = window.confirm("You have unsaved block edits. Revert to default extracted values?");
      if (!confirmed) return;
    }

    setBlockData((prev) => {
      if (!prev?.blocks?.length) return prev;
      const nextBlocks = prev.blocks.map((b, idx) => ({
        ...b,
        weight: clampBlockWeight(Number(extractedBlockWeights[idx]) || 0),
      }));
      return { ...prev, blocks: nextBlocks };
    });
    setOriginalBlockWeights(extractedBlockWeights.map((w) => clampBlockWeight(Number(w) || 0)));
    setActiveWeightsView({ type: "default", label: "Default" });
  }, [blockData, extractedBlockWeights, originalBlockWeights]);

  // ---------------------------
  // Combine Workbench
  // ---------------------------


  const resultsById = useMemo(() => new Map(results.map((r) => [r.stable_id, r])), [results]);

  const combineFirstPick = useMemo(() => {
    if (!combineSelectedIds.length) return null;
    return resultsById.get(combineSelectedIds[0]) ?? null;
  }, [combineSelectedIds, resultsById]);

  const combineCompatibilityKey = useMemo(() => {
    if (!combineFirstPick) return null;
    const base = (combineFirstPick.base_model_code || "").toUpperCase();
    const layout = (combineFirstPick.block_layout || "").toLowerCase();
    if (!base || !layout) return null;
    return `${base}::${layout}`;
  }, [combineFirstPick]);

  const combineCatalog = useMemo(() => {
    const text = combineSearch.trim().toLowerCase();
    const baseList = filteredByLayoutAndSort(results, sortMode, layoutFilter);

    let items = baseList;

    if (text) {
      items = items.filter((it) => (it.filename || "").toLowerCase().includes(text) || (it.stable_id || "").toLowerCase().includes(text));
    }

    // Hide incompatible after first pick unless user toggles showAll
    if (!combineShowAll && combineCompatibilityKey) {
      const [base, layout] = combineCompatibilityKey.split("::");
      items = items.filter((it) => (it.base_model_code || "").toUpperCase() === base && (it.block_layout || "").toLowerCase() === layout);
    }

    return items;
  }, [combineSearch, results, sortMode, layoutFilter, combineShowAll, combineCompatibilityKey]);

  const combineHiddenCount = useMemo(() => {
    if (!combineCompatibilityKey) return 0;
    const all = filteredByLayoutAndSort(results, sortMode, layoutFilter);
    const [base, layout] = combineCompatibilityKey.split("::");
    const compatible = all.filter(
      (it) => (it.base_model_code || "").toUpperCase() === base && (it.block_layout || "").toLowerCase() === layout
    );
    return Math.max(0, all.length - compatible.length);
  }, [results, sortMode, layoutFilter, combineCompatibilityKey]);

  function filteredByLayoutAndSort(items, sort, layout) {
    const sorted = sortLoras(items, sort);
    if (layout === "ALL_LAYOUTS") return sorted;
    return sorted.filter((it) => (it?.block_layout || "").toLowerCase() === layout);
  }

  const combineSelectedItems = useMemo(() => {
    return combineSelectedIds
      .map((id) => resultsById.get(id))
      .filter(Boolean);
  }, [combineSelectedIds, resultsById]);

  const combineComputedById = useMemo(() => {
    const combined = combineResult?.combined;
    if (!combined) return new Map();
    if (Array.isArray(combined)) {
      const m = new Map();
      for (const entry of combined) {
        if (entry?.stable_id) m.set(entry.stable_id, entry);
      }
      return m;
    }
    if (typeof combined === "object") {
      const m = new Map();
      for (const [stableId, val] of Object.entries(combined)) {
        m.set(stableId, { stable_id: stableId, ...(val || {}) });
      }
      return m;
    }
    return new Map();
  }, [combineResult]);

  function handleToggleCombineSelect(stableId) {
    if (!stableId) return;
    setCombineSelectedIds((prev) => (prev.includes(stableId) ? prev.filter((x) => x !== stableId) : [...prev, stableId]));
  }

  function handleRemoveFromStack(stableId) {
    setCombineSelectedIds((prev) => prev.filter((x) => x !== stableId));
  }

  function handleClearCombine() {
    setCombineSelectedIds([]);
    setCombineResult(null);
    setCombineError("");
    setCombineShowAll(false);
  }

  async function handleCalculateCombine() {
    if (!combineSelectedIds.length) return;

    try {
      setCombineLoading(true);
      setCombineError("");

      const res = await fetch(`${API_BASE}/lora/combine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stable_ids: combineSelectedIds }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(bannerString(err.detail ?? err.message ?? err) || `Combine failed (${res.status})`);
      }

      const data = await res.json();
      setCombineResult(data);
    } catch (err) {
      setCombineResult(null);
      setCombineError(bannerString(err) || err?.message || "Failed to calculate combine configuration");
    } finally {
      setCombineLoading(false);
    }
  }

  useEffect(() => {
    setCombineError("");
    setCombineResult(null);
  }, [combineSelectedIds]);

  // ---------------------------
  // Layout options
  // ---------------------------

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

  const apiBaseDisplay = API_BASE.replace("/api", "");

  return (
    <div className="lm-app">
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
              {BASE_MODELS.map((m) => (
                <option key={m.code} value={m.code}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="category">Category</label>
            <select id="category" className="lm-select" value={category} onChange={(e) => setCategory(e.target.value)}>
              {CATEGORIES.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="search">Search</label>
            <input
              id="search"
              className="lm-input"
              type="text"
              placeholder="Filename contains..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <label className="lm-checkbox-row">
            <input type="checkbox" checked={onlyBlocks} onChange={(e) => setOnlyBlocks(e.target.checked)} />
            <span>Only LoRAs with block weights</span>
          </label>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="layout-filter">Block layout</label>
            <select id="layout-filter" className="lm-select" value={layoutFilter} onChange={(e) => setLayoutFilter(e.target.value)}>
              <option value="ALL_LAYOUTS">All layouts</option>
              {layoutOptions.map((layout) => (
                <option key={layout} value={layout}>
                  {formatLayoutLabel(layout)}
                </option>
              ))}
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
              <option value="name_asc">Name · A → Z</option>
              <option value="name_desc">Name · Z → A</option>
              <option value="date_new">Date added · Newest</option>
              <option value="date_old">Date added · Oldest</option>
            </select>
          </div>

          <div className="lm-filter-actions">
            <button type="submit" className="lm-button" disabled={loading} title="Run search with current filters">
              {loading ? "Searching..." : "Run search"}
            </button>
            <button
              type="button"
              className="lm-button lm-button-secondary"
              onClick={handleFullRescan}
              disabled={isRescanning}
              title="Full rescan & reindex all LoRAs"
            >
              {isRescanning ? "Rescanning..." : "Rescan & reindex"}
            </button>
          </div>
        </form>

        {isRescanning && (
          <div className="lm-rescan-progress">
            <div className="lm-rescan-bar">
              <div className="lm-rescan-bar-fill" />
            </div>
            <div className="lm-rescan-text">{rescanProgress?.indexing ? "Indexing in progress..." : "Starting rescan..."}</div>
          </div>
        )}

        <div className="lm-sidebar-footer">
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-badge">
              {totalResults} total · page {currentPage + 1}/{totalPages}
            </span>
          </div>
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-backend">Backend: {apiBaseDisplay}</span>
          </div>
          {lastScanSummary && <div className="lm-last-scan">{lastScanSummary}</div>}
        </div>
      </aside>

      <main className="lm-main">
        <header className="lm-main-header">
          <div>
            <div className="lm-main-pill-row" role="tablist" aria-label="Main views">
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === DASHBOARD_TAB}
                className={classNames("lm-main-pill", activeTab === DASHBOARD_TAB && "lm-main-pill-active")}
                onClick={() => setActiveTab(DASHBOARD_TAB)}
              >
                Dashboard
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === COMBINE_TAB}
                className={classNames("lm-main-pill", activeTab === COMBINE_TAB && "lm-main-pill-active")}
                onClick={() => setActiveTab(COMBINE_TAB)}
              >
                Combine
              </button>
              <button type="button" className="lm-main-pill" disabled>
                Patterns
              </button>
              <button type="button" className="lm-main-pill" disabled>
                Compare
              </button>
              <button type="button" className="lm-main-pill" disabled>
                Delta Lab
              </button>
            </div>
            <div className="lm-main-subtitle">
              {currentBaseLabel} / {currentCategoryLabel} / {onlyBlocks ? "Block-weighted only" : "All LoRAs"}
            </div>
          </div>
        </header>

        {activeTab === DASHBOARD_TAB && (
          <section className="lm-layout">
            <section className="lm-results">
              <div className="lm-results-header">
                <div className="lm-results-title">LoRA catalog</div>
                <div className="lm-results-count">{filteredByLayoutAndSort(results, sortMode, layoutFilter).length} in view · {totalResults} total</div>
              </div>

              {errorMsg && (
                <div className="lm-error-banner">
                  <span>{String(errorMsg)}</span>
                </div>
              )}

              {warningMsg && (
                <div className="lm-warning-banner">
                  <span>{String(warningMsg)}</span>
                </div>
              )}

              {!loading && filteredByLayoutAndSort(results, sortMode, layoutFilter).length === 0 && !errorMsg && (
                <div className="lm-empty-state">No LoRAs match the current filters.</div>
              )}

              {filteredByLayoutAndSort(results, sortMode, layoutFilter).length > 0 && (
                <div className="lm-results-grid">
                  {filteredByLayoutAndSort(results, sortMode, layoutFilter).map((item) => {
                    const hasBlocksFlag = Boolean(item.has_block_weights);
                    const isSelected = item.stable_id && item.stable_id === selectedStableId;
                    return (
                      <article
                        key={item.id}
                        className={classNames("lm-card", isSelected && "lm-card-selected")}
                        onClick={() => handleCardClick(item)}
                        title="Click to view details"
                      >
                        <div className="lm-card-header">
                          <div className="lm-card-id">{item.stable_id || "UNASSIGNED"}</div>
                          <div className={classNames("lm-card-badge", hasBlocksFlag ? "lm-badge-blocks" : "lm-badge-noblocks")}>
                            {getBlocksBadge(item)}
                          </div>
                        </div>
                        <div className="lm-card-filename" title={item.filename || ""}>
                          {item.filename}
                        </div>
                        <div className="lm-card-path">{(item.file_path || "").replace(/\\/g, "/")}</div>
                        <div className="lm-card-footer">
                          <span className="lm-chip">{item.base_model_code}</span>
                          <span className="lm-chip lm-chip-soft">{item.category_code}</span>
                          <span className="lm-chip lm-chip-soft" title={item.block_layout || ""}>
                            {getLayoutBadge(item.block_layout)}
                          </span>
                          <span className="lm-chip lm-chip-type" title={getLoraTypeLabel(item)}>
                            {getTypeBadge(item)}
                          </span>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}

              {totalPages > 1 && (
                <div className="lm-pagination">
                  <button className="lm-page-btn" disabled={currentPage === 0} onClick={() => handlePageChange(0)} title="First page">
                    «
                  </button>
                  <button
                    className="lm-page-btn"
                    disabled={currentPage === 0}
                    onClick={() => handlePageChange(currentPage - 1)}
                    title="Previous page"
                  >
                    ‹
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
                    ›
                  </button>
                  <button
                    className="lm-page-btn"
                    disabled={currentPage >= totalPages - 1}
                    onClick={() => handlePageChange(totalPages - 1)}
                    title="Last page"
                  >
                    »
                  </button>
                </div>
              )}
            </section>

            <section className="lm-details">
              <div className="lm-details-card">
                <header className="lm-details-header">
                  <div className="lm-details-title-block">
                    <div className="lm-details-label">LoRA Details</div>
                    <div className="lm-details-filename">{selectedDetails?.filename || "Select a LoRA"}</div>
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

                {detailsLoading && <div className="lm-details-loading">Loading details...</div>}

                {!detailsLoading && selectedDetails && (
                  <>
                    <dl className="lm-details-grid">
                      <dt>Path</dt>
                      <dd className="lm-dd-path" style={{ paddingLeft: 0 }}>
                        <span title={selectedDetails.file_path}>{selectedDetails.file_path}</span>
                      </dd>
                    </dl>

                    <div className="lm-details-meta">
                      <span>Created: {formatDateOnly(selectedDetails.created_at) || "not recorded"}</span>
                      <span>Updated: {formatDateOnly(selectedDetails.updated_at) || "not recorded"}</span>
                    </div>

                    {hasAnyBlocks && (
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
                          {copyWeightsStatus === "copied"
                            ? "Copied!"
                            : copyWeightsStatus === "failed"
                              ? "Copy failed"
                              : copyWeightsStatus === "copying"
                                ? "Copying..."
                                : "Copy Weights"}
                        </button>
                      </div>
                    )}
                  </>
                )}

                <div className="lm-blocks-panel">
                  <div className="lm-blocks-header">
                    <div className="lm-blocks-title-row">
                      <div className="lm-blocks-title">Block weights</div>
                      {isFallbackBlocks && <span className="lm-fallback-badge" title={fallbackReason}>FALLBACK</span>}
                      <span
                        className="lm-weights-source"
                        title={activeWeightsView.type === "default" ? "Using extracted baseline values" : "Using loaded profile values"}
                      >
                        {activeWeightsView.type === "default" ? "Viewing: Default" : `Viewing: ${activeWeightsView.label}`}
                      </span>
                      <span className="lm-dirty-indicator" style={{ visibility: isDirty ? "visible" : "hidden" }} aria-hidden={!isDirty}>
                        Unsaved changes
                      </span>
                    </div>
                    <div className="lm-blocks-controls">
                      <button
                        className="lm-action-btn lm-action-btn-sm"
                        type="button"
                        onClick={handleUseDefaultWeights}
                        disabled={!hasAnyBlocks || (activeWeightsView.type === "default" && !isDirty)}
                        title="Use extracted default block values"
                      >
                        Default
                      </button>
                      <button
                        className="lm-action-btn lm-action-btn-sm"
                        type="button"
                        onClick={handleResetAllBlocks}
                        disabled={!isDirty}
                        title="Reset all blocks to originally loaded DB values"
                      >
                        Reset all
                      </button>
                      <div className="lm-blocks-count">{hasAnyBlocks ? `${blockData.blocks.length} blocks` : "No block weights"}</div>
                    </div>
                  </div>

                  {hasAnyBlocks && (
                    <BlockPanelErrorBoundary>
                      <div className={classNames("lm-blocks-list", isFallbackBlocks && "lm-blocks-list-fallback", compactMode && "lm-blocks-list-compact")}>
                        <div className="lm-blocks-analytics">
                          <span>min {blockStats.min.toFixed(1)}</span>
                          <span>max {blockStats.max.toFixed(1)}</span>
                          <span>mean {blockStats.mean.toFixed(1)}</span>
                          <span>var {blockStats.variance.toFixed(3)}</span>
                        </div>
                        {blockData.blocks.map((b) => (
                          <BlockRow
                            key={b.block_index}
                            block={b}
                            compactMode={compactMode}
                            showSlider={effectiveShowSliders}
                            isFallbackBlocks={isFallbackBlocks}
                            isDirty={Boolean(dirtyByIndex[b.block_index])}
                            onWeightChange={handleBlockWeightChange}
                            onReset={handleResetSingleBlock}
                          />
                        ))}
                      </div>
                    </BlockPanelErrorBoundary>
                  )}

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

                {selectedDetails && (
                  <div className="lm-profiles-panel">
                    <div className="lm-profiles-header">
                      <div className="lm-profiles-title">Saved profiles</div>
                      <div className="lm-profiles-count">
                        {profilesLoading ? "Loading..." : `${profiles.length} profile${profiles.length === 1 ? "" : "s"}`}
                      </div>
                    </div>

                    {hasAnyBlocks && (
                      <div className="lm-profile-save-row">
                        <input
                          className="lm-input lm-profile-name-input"
                          type="text"
                          placeholder={editingProfileId ? "Edit profile name..." : "Profile name..."}
                          value={editingProfileId ? editingProfileName : newProfileName}
                          onChange={(e) => (editingProfileId ? setEditingProfileName(e.target.value) : setNewProfileName(e.target.value))}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") editingProfileId ? handleUpdateProfile() : handleSaveProfile();
                          }}
                        />
                        {editingProfileId ? (
                          <>
                            <button className="lm-action-btn" onClick={handleUpdateProfile} disabled={savingProfile || !editingProfileName.trim()}>
                              {savingProfile ? "Updating..." : "Update"}
                            </button>
                            <button className="lm-action-btn lm-action-btn-sm" onClick={handleCancelEdit} disabled={savingProfile}>
                              Cancel
                            </button>
                          </>
                        ) : (
                          <button className="lm-action-btn" onClick={handleSaveProfile} disabled={savingProfile || !newProfileName.trim()}>
                            {savingProfile ? "Saving..." : "Save"}
                          </button>
                        )}
                      </div>
                    )}

                    {profiles.length > 0 && (
                      <div className="lm-profile-list">
                        {profiles.map((p) => (
                          <div className="lm-profile-item" key={p.id}>
                            <div className="lm-profile-item-info">
                              <div className="lm-profile-item-name">{p.profile_name}</div>
                              <div className="lm-profile-item-meta">{p.block_weights?.length ?? 0} blocks · {p.updated_at}</div>
                            </div>
                            <div className="lm-profile-item-actions">
                              <button className="lm-action-btn lm-action-btn-sm" onClick={() => handleLoadProfile(p)} title="Load this profile into view">
                                Load
                              </button>
                              <button className="lm-action-btn lm-action-btn-sm" onClick={() => handleEditProfile(p)} title="Edit this profile">
                                Edit
                              </button>
                              <button
                                className="lm-action-btn lm-action-btn-sm lm-action-btn-danger"
                                onClick={() => handleDeleteProfile(p.id)}
                                title="Delete this profile"
                              >
                                Del
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {!profilesLoading && profiles.length === 0 && <div className="lm-profiles-empty">No saved profiles yet.</div>}
                  </div>
                )}
              </div>
            </section>
          </section>
        )}

        {activeTab === COMBINE_TAB && (
          <section className="lm-combine-workbench">
            <div className="lm-combine-toolbar">
              <div className="lm-combine-toolbar-left">
                <div className="lm-results-title">Combine workbench</div>
                <div className="lm-results-count">Selected: {combineSelectedIds.length}</div>
              </div>

              <div className="lm-combine-toolbar-right">
                
                {combineCompatibilityKey && (
                  <button
                    type="button"
                    className="lm-action-btn lm-action-btn-sm"
                    onClick={() => setCombineShowAll((v) => !v)}
                    title="Toggle hiding incompatible cards"
                  >
                    Showing: {combineShowAll ? "All" : "Compatible"}
                    {!combineShowAll && combineHiddenCount > 0 ? ` (hiding ${combineHiddenCount})` : ""}
                  </button>
                )}

                <button
                  type="button"
                  className="lm-action-btn lm-action-btn-sm"
                  onClick={handleClearCombine}
                  disabled={combineSelectedIds.length === 0 && !combineResult && !combineError}
                  title="Clear selection and results"
                >
                  Clear
                </button>

                <button
                  type="button"
                  className="lm-action-btn lm-action-btn-sm lm-combine-calc"
                  disabled={!combineSelectedIds.length || combineLoading}
                  onClick={handleCalculateCombine}
                >
                  {combineLoading ? "Calculating..." : "Calculate configuration"}
                </button>
              </div>
            </div>

            {combineError && (
              <div className="lm-error-banner">
                <span>{bannerString(combineError)}</span>
              </div>
            )}

            {Array.isArray(combineResult?.warnings) && combineResult.warnings.length > 0 && (
              <div className="lm-warning-banner">Warnings: {bannerString(combineResult.warnings)}</div>
            )}

            {Array.isArray(combineResult?.excluded_loras) && combineResult.excluded_loras.length > 0 && (
              <div className="lm-warning-banner">Excluded: {bannerString(combineResult.excluded_loras)}</div>
            )}

            <div className="lm-combine-columns">
              <section className="lm-combine-catalog">
                <div className="lm-combine-catalog-header">
                  <div className="lm-combine-catalog-title">Catalog</div>
                  <input
                    className="lm-input"
                    value={combineSearch}
                    onChange={(e) => setCombineSearch(e.target.value)}
                    placeholder="Search catalog in Combine..."
                    title="Search by stable id or filename"
                  />
                </div>

                <div className="lm-results-grid">
                  {combineCatalog.map((item) => {
                    const isPicked = combineSelectedIds.includes(item.stable_id);
                    const hasBlocksFlag = Boolean(item.has_block_weights);
                    return (
                      <article
                        key={item.id}
                        className={classNames(
                          "lm-card",
                          isPicked && "lm-card-multi-selected",
                          "lm-card-selectable"
                        )}
                        onClick={() => {
                          handleToggleCombineSelect(item.stable_id);
                        }}
                        title={isPicked ? "Click to remove" : "Click to add"}
                      >
                        <div className="lm-card-header">
                          <div className="lm-card-id">{item.stable_id || "UNASSIGNED"}</div>
                          <div className={classNames("lm-card-badge", hasBlocksFlag ? "lm-badge-blocks" : "lm-badge-noblocks")}>
                            {getBlocksBadge(item)}
                          </div>
                        </div>
                        <div className="lm-card-filename" title={item.filename || ""}>
                          {item.filename}
                        </div>
                        <div className="lm-card-path">{(item.file_path || "").replace(/\\/g, "/")}</div>
                        <div className="lm-card-footer">
                          <span className="lm-chip">{item.base_model_code}</span>
                          <span className="lm-chip lm-chip-soft">{item.category_code}</span>
                          <span className="lm-chip lm-chip-soft" title={item.block_layout || ""}>
                            {getLayoutBadge(item.block_layout)}
                          </span>
                          <span className="lm-chip lm-chip-type" title={getLoraTypeLabel(item)}>
                            {getTypeBadge(item)}
                          </span>
                        </div>
                      </article>
                    );
                  })}

                  {combineCatalog.length === 0 && <div className="lm-empty-state">No catalog items match your Combine search/filters.</div>}
                </div>
              </section>

              <section className="lm-combine-stack">
                <div className="lm-combine-stack-header">
                  <div className="lm-combine-catalog-title">Selected stack</div>
                  <div className="lm-results-count">{combineSelectedIds.length ? "Click × to remove" : "Pick some cards from the left"}</div>
                </div>

                <div className="lm-combine-stack-list">
                  {combineSelectedItems.map((item) => {
                    const computed = combineComputedById.get(item.stable_id) || {};
                    const blockList = computed.block_weights || computed.block_weight_list || computed.blocks || null;
                    const blockCsv = Array.isArray(blockList) ? blockList.map((v) => Number(v).toFixed(1)).join(",") : "";
                    const hasComputed = Boolean(computed && Object.keys(computed).length);

                    return (
                      <article key={item.stable_id} className={classNames("lm-combine-card", hasComputed && "lm-combine-card-computed")}>
                        <div className="lm-combine-card-top">
                          <div>
                            <div className="lm-card-id">{item.stable_id}</div>
                            <div className="lm-card-filename" title={item.filename || ""}>{item.filename}</div>
                          </div>
                          <button
                            type="button"
                            className="lm-action-btn lm-action-btn-sm lm-action-btn-danger"
                            onClick={() => handleRemoveFromStack(item.stable_id)}
                            title="Remove from stack"
                          >
                            ×
                          </button>
                        </div>

                        <div className="lm-combine-metrics">
                          <div className="lm-combine-metric"><span>strength_model</span><b>{computed.strength_model ?? "-"}</b></div>
                          <div className="lm-combine-metric"><span>strength_clip</span><b>{computed.strength_clip ?? "-"}</b></div>
                          <div className="lm-combine-metric"><span>A</span><b>{computed.A ?? "-"}</b></div>
                          <div className="lm-combine-metric"><span>B</span><b>{computed.B ?? "-"}</b></div>
                        </div>

                        <div className="lm-combine-weights">
                          <div className="lm-combine-weights-row">
                            <span>block weights</span>
                            <b>{Array.isArray(blockList) ? blockList.length : "-"}</b>
                          </div>
                          <div className="lm-combine-weights-actions">
                            <CopyButton text={blockCsv} label="Copy weights" />
                          </div>
                          <div className="lm-combine-preview" title={blockCsv || ""}>
                            {blockCsv ? `preview: ${blockCsv.slice(0, 120)}${blockCsv.length > 120 ? "…" : ""}` : "preview: -"}
                          </div>
                        </div>
                      </article>
                    );
                  })}

                  {combineSelectedItems.length === 0 && (
                    <div className="lm-empty-state">Select LoRAs in the left catalog.</div>
                  )}
                </div>
              </section>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
