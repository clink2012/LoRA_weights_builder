import os
import json
from typing import Dict, Any, List, Tuple

# This must match your profile_builder config
PROFILES_ROOT = r"E:\models\loras\Database\backend\profiles"

# In-memory cache
_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}
_PROFILE_LIST: List[Dict[str, Any]] = []


def _is_profile_file(name: str) -> bool:
    return name.lower().endswith("_profile.json")


def _scan_profile_files() -> List[str]:
    """
    Walk PROFILES_ROOT and return a list of full paths
    to *_profile.json files.
    """
    paths: List[str] = []

    if not os.path.isdir(PROFILES_ROOT):
        return paths

    for root, dirs, files in os.walk(PROFILES_ROOT):
        for fn in files:
            if _is_profile_file(fn):
                full = os.path.join(root, fn)
                paths.append(full)

    return paths


def _make_profile_id(profile: Dict[str, Any]) -> str:
    """
    Decide how to identify a profile in the API / UI.

    For now: use the file 'stem' (without _profile suffix) plus bucket.
    Example: flux/EmmaWatson_flux_lora_v1
    """
    file_info = profile.get("file", {})
    model_info = profile.get("model", {})

    stem = file_info.get("stem") or file_info.get("name") or "unknown"
    bucket = model_info.get("bucket") or "unknown"

    return f"{bucket}/{stem}"


def reload_profiles() -> Tuple[int, int]:
    """
    Reload all profiles from disk into memory.

    Returns:
        (loaded_count, error_count)
    """
    global _PROFILE_CACHE, _PROFILE_LIST

    new_cache: Dict[str, Dict[str, Any]] = {}
    new_list: List[Dict[str, Any]] = []

    paths = _scan_profile_files()
    errors = 0

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            errors += 1
            continue

        # Ensure a profile_id so the UI can address it
        pid = profile.get("profile_id")
        if not pid:
            pid = _make_profile_id(profile)
            profile["profile_id"] = pid

        new_cache[pid] = profile
        new_list.append(profile)

    _PROFILE_CACHE = new_cache
    _PROFILE_LIST = new_list

    return len(new_list), errors


def get_all_profiles() -> List[Dict[str, Any]]:
    """
    Return a list of all profile dicts currently in memory.
    Safe to use directly in the web API.
    """
    return list(_PROFILE_LIST)


def get_profile(profile_id: str) -> Dict[str, Any]:
    """
    Look up a single profile by its profile_id.
    Raises KeyError if not found.
    """
    if profile_id not in _PROFILE_CACHE:
        raise KeyError(f"No profile with id '{profile_id}'")
    return _PROFILE_CACHE[profile_id]


def ensure_loaded_once() -> None:
    """
    Call this once at startup (or lazily on first request)
    to ensure profiles are loaded.
    """
    if not _PROFILE_CACHE:
        reload_profiles()


if __name__ == "__main__":
    # Simple manual test harness
    print(f"Scanning profiles under: {PROFILES_ROOT}")
    count, errors = reload_profiles()
    print(f"Loaded {count} profile(s), errors: {errors}")
    if count:
        print("Example profile_id values:")
        for p in _PROFILE_LIST[:5]:
            print(" -", p.get("profile_id"))
