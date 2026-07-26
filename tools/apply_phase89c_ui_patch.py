from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "Database" / "UI" / "src" / "App.jsx"
TEST_PATH = REPO_ROOT / "Database" / "UI" / "src" / "App.smoke.test.jsx"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} fragment, found {count}.")
    return text.replace(old, new, 1)


def patch_app(text: str) -> str:
    old_models = '''const BASE_MODELS = [
  { code: "FLX", label: "Flux" },
  { code: "FLK", label: "Flux Krea" },
  { code: "W21", label: "WAN 2.1" },
  { code: "W22", label: "WAN 2.2" },
  { code: "PNY", label: "Pony" },
  { code: "SDX", label: "SDXL" },
  { code: "SD1", label: "SD 1.x" },
  { code: "ILL", label: "Illustrious" },
  { code: "ALL", label: "All Models" },
];'''
    new_models = '''const FALLBACK_BASE_MODELS = [
  { code: "FLX", label: "Flux", supportLevel: "mixed-scanned-fallback" },
  { code: "FLK", label: "Flux Krea", supportLevel: "mixed-scanned-fallback" },
  { code: "F2K", label: "Flux.2-Klein · metadata only", supportLevel: "metadata-only" },
  { code: "ILL", label: "Illustrious · metadata only", supportLevel: "metadata-only" },
  { code: "LTX", label: "LTXV2 · metadata only", supportLevel: "metadata-only" },
  { code: "PNY", label: "Pony · metadata only", supportLevel: "metadata-only" },
  { code: "SD1", label: "SD 1.x · metadata only", supportLevel: "metadata-only" },
  { code: "SDX", label: "SDXL · metadata only", supportLevel: "metadata-only" },
  { code: "W21", label: "WAN 2.1 · metadata only", supportLevel: "metadata-only" },
  { code: "W22", label: "WAN 2.2 · metadata only", supportLevel: "metadata-only" },
  { code: "ZIM", label: "Z-Image · metadata only", supportLevel: "metadata-only" },
  { code: "ALL", label: "All Models", supportLevel: "all" },
];

function modelFamilyOptionsFromApi(payload) {
  const families = Array.isArray(payload?.families) ? payload.families : [];
  const options = families
    .map((family) => {
      const code = String(family?.code || "").trim().toUpperCase();
      const displayName = String(family?.display_name || "").trim();
      const supportLevel = String(family?.support_level || "").trim();
      if (!code || !displayName) return null;
      const suffix = supportLevel === "metadata-only" ? " · metadata only" : "";
      return { code, label: `${displayName}${suffix}`, supportLevel };
    })
    .filter(Boolean);

  if (!options.length) return FALLBACK_BASE_MODELS;
  return [...options, { code: "ALL", label: "All Models", supportLevel: "all" }];
}'''
    text = replace_once(text, old_models, new_models, "base-model constant")

    old_state = '''function App() {
  const [baseModel, setBaseModel] = useState("FLX");
  const [category, setCategory] = useState("ALL");'''
    new_state = '''function App() {
  const [baseModel, setBaseModel] = useState("FLX");
  const [baseModels, setBaseModels] = useState(FALLBACK_BASE_MODELS);
  const [category, setCategory] = useState("ALL");'''
    text = replace_once(text, old_state, new_state, "base-model state")

    old_effect = '''  useEffect(() => {
    runSearch(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!isRescanning) {'''
    new_effect = '''  useEffect(() => {
    let cancelled = false;

    async function loadModelFamilies() {
      try {
        const res = await fetch(`${API_BASE}/model-families`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) setBaseModels(modelFamilyOptionsFromApi(data));
      } catch {
        // Keep the complete fallback registry during staggered deployments.
      }
    }

    loadModelFamilies();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    runSearch(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!isRescanning) {'''
    text = replace_once(text, old_effect, new_effect, "initial effects")

    text = replace_once(
        text,
        '  const currentBaseLabel = BASE_MODELS.find((b) => b.code === baseModel)?.label || "Unknown";',
        '  const currentBaseLabel = baseModels.find((b) => b.code === baseModel)?.label || "Unknown";',
        "current base label",
    )
    text = replace_once(text, "{BASE_MODELS.map((m) => (", "{baseModels.map((m) => (", "base-model options render")
    return text


def patch_test(text: str) -> str:
    old_mock_anchor = '''      const url = String(input);

      if (url.includes("/lora/search")) {'''
    new_mock_anchor = '''      const url = String(input);

      if (url.endsWith("/model-families")) {
        return jsonResponse({
          schema_version: "8.9a",
          families: [
            { code: "FLX", display_name: "Flux", support_level: "mixed-scanned-fallback" },
            { code: "F2K", display_name: "Flux.2-Klein", support_level: "metadata-only" },
            { code: "LTX", display_name: "LTXV2", support_level: "metadata-only" },
            { code: "ZIM", display_name: "Z-Image", support_level: "metadata-only" },
          ],
        });
      }

      if (url.includes("/lora/search")) {'''
    text = replace_once(text, old_mock_anchor, new_mock_anchor, "model-family API mock")

    old_test_anchor = '''  it("renders stack health for structured combine errors", async () => {'''
    new_test = '''  it("loads model families from the backend registry and labels metadata-only options", async () => {
    render(<App />);

    const select = screen.getByLabelText("Base model");
    await waitFor(() => {
      const labels = Array.from(select.options).map((option) => option.textContent);
      expect(labels).toContain("Flux");
      expect(labels).toContain("Flux.2-Klein · metadata only");
      expect(labels).toContain("LTXV2 · metadata only");
      expect(labels).toContain("Z-Image · metadata only");
      expect(labels).toContain("All Models");
    });

    expect(globalThis.fetch).toHaveBeenCalledWith(expect.stringContaining("/model-families"));
  });

  it("renders stack health for structured combine errors", async () => {'''
    text = replace_once(text, old_test_anchor, new_test, "model-family UI test")
    return text


def main() -> int:
    app_before = APP_PATH.read_text(encoding="utf-8")
    test_before = TEST_PATH.read_text(encoding="utf-8")

    app_after = patch_app(app_before)
    test_after = patch_test(test_before)

    APP_PATH.write_text(app_after, encoding="utf-8")
    TEST_PATH.write_text(test_after, encoding="utf-8")

    print(f"Updated: {APP_PATH.relative_to(REPO_ROOT)}")
    print(f"Updated: {TEST_PATH.relative_to(REPO_ROOT)}")
    print("The model-family endpoint now drives the dropdown with a complete fallback registry.")

    Path(__file__).unlink()
    print(f"Removed one-shot helper: {Path(__file__).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
