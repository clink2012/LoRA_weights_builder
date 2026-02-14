import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5001/api";

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

  // Use numeric id as a proxy for "added order"
  if (mode === "date_new" || mode === "date_old") {
    data.sort((a, b) => {
      const ai = a.id ?? 0;
      const bi = b.id ?? 0;
      return ai - bi;
    });
    if (mode === "date_new") data.reverse();
    return data;
  }

  return data;
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

function getDisplayBlockCount(item) {
  const hasBlocks = Boolean(item?.has_block_weights);
  const expected = getExpectedBlocksFromLayout(item?.block_layout);

  if (hasBlocks) {
    if (expected !== null) {
      return `${expected} blocks`;
    }
    const actual = Number.isFinite(item?.block_count) ? item.block_count : null;
    if (actual !== null) {
      return `${actual} blocks`;
    }
    return "Blocks";
  }

  const baseCode = (item?.base_model_code || "").toUpperCase();
  if ((baseCode === "FLX" || baseCode === "FLK") && item?.block_layout === "flux_fallback_16") {
    return "16 blocks";
  }

  return "No blocks";
}

function App() {
  const [baseModel, setBaseModel] = useState("FLX");
  const [category, setCategory] = useState("ALL");
  const [search, setSearch] = useState("");
  const [onlyBlocks, setOnlyBlocks] = useState(false);
  const [layoutFilter, setLayoutFilter] = useState("ALL_LAYOUTS");

  const [results, setResults] = useState([]);
  const [resultsCount, setResultsCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const [selectedStableId, setSelectedStableId] = useState(null);
  const [selectedDetails, setSelectedDetails] = useState(null);
  const [blockData, setBlockData] = useState(null);
  const [detailsLoading, setDetailsLoading] = useState(false);

  const [sortMode, setSortMode] = useState("name_asc");
  const [lastScanSummary, setLastScanSummary] = useState("");
  const [isRescanning, setIsRescanning] = useState(false);

  // Initial search on first load
  useEffect(() => {
    runSearch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentBaseLabel =
    BASE_MODELS.find((b) => b.code === baseModel)?.label || "Unknown";
  const currentCategoryLabel =
    CATEGORIES.find((c) => c.code === category)?.label || "Unknown";

  async function runSearch() {
    try {
      setLoading(true);
      setErrorMsg("");
      setResults([]);
      setSelectedStableId(null);
      setSelectedDetails(null);
      setBlockData(null);

      const params = new URLSearchParams();

      if (baseModel && baseModel !== "ALL") {
        params.set("base", baseModel);
      }
      if (category && category !== "ALL") {
        params.set("category", category);
      }
      if (search.trim()) {
        params.set("search", search.trim());
      }
      if (onlyBlocks) {
        params.set("has_blocks", "1");
      }
      params.set("limit", "2000");

      const url = `${API_BASE}/lora/search?${params.toString()}`;
      console.log("Running search:", url);

      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Search failed with status ${res.status}`);
      }

      const data = await res.json();

      let list = [];
      if (Array.isArray(data)) {
        list = data;
      } else if (Array.isArray(data.results)) {
        list = data.results;
      } else {
        console.warn("Unexpected search response shape:", data);
      }

      const withBlockCount = list.map((item) => ({
        ...item,
        block_count: Number.isFinite(item?.block_count)
          ? item.block_count
          : item.has_block_weights
            ? getExpectedBlocksFromLayout(item.block_layout)
            : 0,
      }));

      const sorted = sortLoras(withBlockCount, sortMode);
      console.log("Search returned", sorted.length, "items");
      setResults(sorted);
      setResultsCount(sorted.length);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleCardClick(item) {
    if (!item || !item.stable_id) return;

    const stableId = item.stable_id;
    try {
      setSelectedStableId(stableId);
      setSelectedDetails(null);
      setBlockData(null);
      setDetailsLoading(true);

      const [detailsRes, blocksRes] = await Promise.all([
        fetch(`${API_BASE}/lora/${stableId}`),
        fetch(`${API_BASE}/lora/${stableId}/blocks`),
      ]);

      if (!detailsRes.ok) {
        throw new Error(`Details request failed (${detailsRes.status})`);
      }

      const detailsJson = await detailsRes.json();
      setSelectedDetails(detailsJson);

      if (blocksRes.ok) {
        const blocksJson = await blocksRes.json();
        setBlockData(blocksJson);
      } else {
        setBlockData(null);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Failed to load details");
    } finally {
      setDetailsLoading(false);
    }
  }

  function handleSearchSubmit(e) {
    e.preventDefault();
    runSearch();
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
      setErrorMsg("");

      const res = await fetch(`${API_BASE}/lora/reindex_all`, {
        method: "POST",
      });

      if (!res.ok) {
        throw new Error(`Reindex failed with status ${res.status}`);
      }

      const info = await res.json();
      console.log("Reindex complete:", info);

      const s = info.summary || {};
      const total = s.total ?? 0;
      const withBlocks = s.with_blocks ?? 0;
      const noBlocks = s.no_blocks ?? 0;
      const duration = info.duration_sec ?? 0;

      const summaryText = `Indexed ${total} LoRAs · With blocks: ${withBlocks} · No blocks: ${noBlocks} · ${duration}s`;
      setLastScanSummary(summaryText);

      await runSearch();
    } catch (err) {
      console.error(err);
      setLastScanSummary("Rescan failed – check backend logs.");
      window.alert(err.message || "Reindex failed – see backend console.");
    } finally {
      setIsRescanning(false);
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
    if (!layoutOptions.includes(layoutFilter)) {
      setLayoutFilter("ALL_LAYOUTS");
    }
  }, [layoutFilter, layoutOptions]);

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
          {/* Base model */}
          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="base-model">
              Base model
            </label>
            <select
              id="base-model"
              className="lm-select"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
            >
              {BASE_MODELS.map((m) => (
                <option key={m.code} value={m.code}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {/* Category */}
          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="category">
              Category
            </label>
            <select
              id="category"
              className="lm-select"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {CATEGORIES.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>

          {/* Search text */}
          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="search">
              Search
            </label>
            <input
              id="search"
              className="lm-input"
              type="text"
              placeholder="Filename contains..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          {/* Only blocks toggle */}
          <label className="lm-checkbox-row">
            <input
              type="checkbox"
              checked={onlyBlocks}
              onChange={(e) => setOnlyBlocks(e.target.checked)}
            />
            <span>Only LoRAs with block weights</span>
          </label>

          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="layout-filter">
              Block layout
            </label>
            <select
              id="layout-filter"
              className="lm-select"
              value={layoutFilter}
              onChange={(e) => setLayoutFilter(e.target.value)}
            >
              <option value="ALL_LAYOUTS">All layouts</option>
              {layoutOptions.map((layout) => (
                <option key={layout} value={layout}>
                  {formatLayoutLabel(layout)}
                </option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div className="lm-filter-group">
            <label className="lm-filter-label" htmlFor="sort-mode">
              Sort
            </label>
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

          {/* Buttons */}
          <div className="lm-filter-actions">
            <button
              type="submit"
              className="lm-button"
              disabled={loading}
              title="Run search with current filters"
            >
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

        {/* Sidebar footer – results + backend + last scan summary */}
        <div className="lm-sidebar-footer">
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-badge">
              {resultsCount} result{resultsCount === 1 ? "" : "s"}
            </span>
          </div>
          <div className="lm-sidebar-footer-row">
            <span className="lm-sidebar-backend">
              Backend: {apiBaseDisplay}
            </span>
          </div>
          {lastScanSummary && (
            <div className="lm-last-scan">{lastScanSummary}</div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="lm-main">
        <header className="lm-main-header">
          <div>
            <div className="lm-main-pill-row">
              <span className="lm-main-pill lm-main-pill-active">
                Dashboard
              </span>
              <span className="lm-main-pill">Patterns</span>
              <span className="lm-main-pill">Compare</span>
              <span className="lm-main-pill">Delta Lab</span>
            </div>
            <div className="lm-main-subtitle">
              {currentBaseLabel} / {currentCategoryLabel} /{" "}
              {onlyBlocks ? "Block-weighted only" : "All LoRAs"}
            </div>
          </div>
        </header>

        <section className="lm-layout">
          {/* Search results */}
          <section className="lm-results">
            <div className="lm-results-header">
              <div className="lm-results-title">LoRA catalog</div>
              <div className="lm-results-count">
                {filteredResults.length} in view
              </div>
            </div>

            {errorMsg && (
              <div className="lm-error-banner">
                <span>{errorMsg}</span>
              </div>
            )}

            {!loading && filteredResults.length === 0 && !errorMsg && (
              <div className="lm-empty-state">
                No LoRAs match the current filters.
              </div>
            )}

            {filteredResults.length > 0 && (
              <div className="lm-results-grid">
                {filteredResults.map((item) => {
                  const hasBlocks = Boolean(item.has_block_weights);
                  const isSelected =
                    item.stable_id && item.stable_id === selectedStableId;

                  const catLabel =
                    CATEGORIES.find((c) => c.code === item.category_code)
                      ?.label || item.category_name;

                  const nicePath = (item.file_path || "").replace(/\\/g, "/");
                  const blockCountLabel = getDisplayBlockCount(item);

                  return (
                    <article
                      key={item.id}
                      className={classNames(
                        "lm-card",
                        isSelected && "lm-card-selected"
                      )}
                      onClick={() => handleCardClick(item)}
                    >
                      <div className="lm-card-header">
                        <div className="lm-card-id">
                          {item.stable_id || "UNASSIGNED"}
                        </div>
                        <div
                          className={classNames(
                            "lm-card-badge",
                            hasBlocks ? "lm-badge-blocks" : "lm-badge-noblocks"
                          )}
                        >
                          {hasBlocks ? "BLOCKS" : "NO BLOCKS"}
                        </div>
                      </div>

                      <div className="lm-card-filename">{item.filename}</div>
                      <div className="lm-card-path">{nicePath}</div>

                      <div className="lm-card-footer">
                        <span className="lm-chip">
                          {item.base_model_code}/{item.category_code}
                        </span>
                        <span className="lm-chip lm-chip-soft">{catLabel}</span>
                        <span className="lm-chip lm-chip-soft">{blockCountLabel}</span>
                        <span className="lm-chip lm-chip-type">{getLoraTypeLabel(item)}</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>

          {/* Details panel */}
          <section className="lm-details">
            <div className="lm-details-card">
              <header className="lm-details-header">
                <div className="lm-details-title-block">
                  <div className="lm-details-label">LoRA Details</div>
                  <div className="lm-details-filename">
                    {selectedDetails?.filename || "Select a LoRA"}
                  </div>
                </div>
                {selectedDetails?.stable_id && (
                  <div className="lm-details-id-stack">
                    <div className="lm-details-stable-id">{selectedDetails.stable_id}</div>
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
                    <div className="lm-details-row">
                      <dt>Filename</dt>
                      <dd>{selectedDetails.filename}</dd>
                    </div>
                    <div className="lm-details-row">
                      <dt>Base model</dt>
                      <dd>
                        {selectedDetails.base_model_code} ·{" "}
                        {selectedDetails.base_model_name}
                      </dd>
                    </div>
                    <div className="lm-details-row">
                      <dt>Category</dt>
                      <dd>
                        {selectedDetails.category_code} ·{" "}
                        {selectedDetails.category_name}
                      </dd>
                    </div>
                    <div className="lm-details-row">
                      <dt>Family</dt>
                      <dd>{selectedDetails.model_family || "Unknown"}</dd>
                    </div>
                    <div className="lm-details-row">
                      <dt>LoRA type</dt>
                      <dd>{getLoraTypeLabel(selectedDetails)}</dd>
                    </div>
                    <div className="lm-details-row">
                      <dt>Rank</dt>
                      <dd>{selectedDetails.rank ?? "Unknown"}</dd>
                    </div>
                  </dl>

                  <div className="lm-details-meta">
                    <span>
                      Created: {selectedDetails.created_at || "not recorded"}
                    </span>
                    <span>
                      Updated: {selectedDetails.updated_at || "not recorded"}
                    </span>
                  </div>
                </>
              )}

              {/* Block weights */}
              <div className="lm-blocks-panel">
                <div className="lm-blocks-header">
                  <div className="lm-blocks-title-row">
                    <div className="lm-blocks-title">Block weights</div>
                    {isFallbackBlocks && (
                      <span className="lm-fallback-badge" title={fallbackReason}>
                        FALLBACK
                      </span>
                    )}
                  </div>
                  <div className="lm-blocks-count">
                    {hasAnyBlocks ? `${blockData.blocks.length} blocks` : "No block weights"}
                  </div>
                </div>

                {isFallbackBlocks && selectedDetails && (
                  <div className="lm-fallback-note" title={fallbackReason}>
                    {fallbackReason}
                  </div>
                )}

                {hasAnyBlocks && (
                  <div
                    className={classNames(
                      "lm-blocks-list",
                      isFallbackBlocks && "lm-blocks-list-fallback"
                    )}
                  >
                    {blockData.blocks.map((b) => (
                      <div
                        className={classNames(
                          "lm-block-row",
                          isFallbackBlocks && "lm-block-row-fallback"
                        )}
                        key={b.block_index}
                      >
                        <div className="lm-block-index">
                          #{String(b.block_index ?? 0).padStart(2, "0")}
                        </div>
                        <div className="lm-block-bar-wrap">
                          <div className="lm-block-bar-bg">
                            <div
                              className="lm-block-bar-fill"
                              style={{
                                width: `${Math.max(2, (Number(b.weight) || 0) * 100).toFixed(1)}%`,
                              }}
                            />
                          </div>
                        </div>
                        <div className="lm-block-value">{(Number(b.weight) || 0).toFixed(3)}</div>
                      </div>
                    ))}
                  </div>
                )}

                {!detailsLoading && selectedDetails && !hasAnyBlocks && (
                  <div className="lm-blocks-empty">
                    {isFallbackBlocks
                      ? "Showing fallback block profile (not extracted weights)."
                      : "This LoRA has no recorded block weights in the database."}
                  </div>
                )}

                {!selectedDetails && !detailsLoading && (
                  <div className="lm-blocks-empty">
                    Select a LoRA card to view its block profile.
                  </div>
                )}
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

export default App;
