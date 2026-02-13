import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple


# --- CONFIG: base model and category mappings --- #

# Map folder name under the root to (short_code, human_readable_name)
BASE_MODEL_MAP: Dict[str, Tuple[str, str]] = {
    "FLUX": ("FLX", "Flux"),
    "Flux Krea": ("FLK", "Flux Krea"),
    "Illustrious": ("ILL", "Illustrious"),
    "PONY": ("PNY", "Pony"),
    "SD": ("SD1", "SD"),
    "SDXL": ("SDX", "SDXL"),
    "WAN2.1": ("W21", "WAN2.1"),
    "WAN2.2": ("W22", "WAN2.2"),
}

# Map the two-digit category index to (short_code, human_readable_name)
CATEGORY_INDEX_MAP: Dict[str, Tuple[str, str]] = {
    "01": ("PPL", "People"),
    "02": ("STL", "Styles"),
    "03": ("UTL", "Utils"),
    "04": ("ACT", "Action"),
    "05": ("BDY", "Body"),
    "06": ("CHT", "Characters"),
    "07": ("MCV", "Machines_Vehicles"),
    "08": ("CLT", "Clothing"),
    "09": ("ANM", "Animals"),
    "10": ("BLD", "Buildings"),
    "11": ("NAT", "Nature"),
}


@dataclass
class LoraRecord:
    file_path: str
    filename: str

    base_model_name: Optional[str]
    base_model_code: Optional[str]

    category_name: Optional[str]
    category_code: Optional[str]

    # These will be filled later when we integrate Delta Inspector and the DB
    lora_type: Optional[str] = None
    rank: Optional[int] = None
    block_weights: Optional[List[float]] = None

    # Placeholder for future: Clink-specific overrides, notes, etc.
    clink_profile_name: Optional[str] = None


def normalise_path(path: str) -> str:
    """Return a normalised, absolute path with consistent separators."""
    return os.path.abspath(os.path.normpath(path))


def parse_base_and_category(
    file_path: str,
    root_dir: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Given a full file path and the root LoRA directory, extract:

    - base_model_name
    - base_model_code
    - category_name
    - category_code

    based on the directory structure:
      root_dir / <BASE_MODEL> / "<NN - Category Name>" / file.safetensors
    """
    root_dir = normalise_path(root_dir)
    file_path_norm = normalise_path(file_path)

    # Make sure the file is under the root directory
    try:
        rel_path = os.path.relpath(file_path_norm, root_dir)
    except ValueError:
        # Different drive, etc.
        return None, None, None, None

    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        # We expect at least: <BASE_MODEL>/<CATEGORY_FOLDER>/<file>
        return None, None, None, None

    base_model_folder = parts[0]
    category_folder = parts[1]

    # --- Base model --- #
    base_model_name = base_model_folder
    base_model_code = None

    if base_model_folder in BASE_MODEL_MAP:
        base_model_code, base_model_human = BASE_MODEL_MAP[base_model_folder]
        # Prefer the mapped human-readable name if desired
        base_model_name = base_model_human
    else:
        # Unknown base model – we still keep the folder name as "name"
        base_model_code = None

    # --- Category --- #
    # Category folder is expected like "03 - Utils" or "08 - Clothing"
    category_code = None
    category_name = None

    # Get the first "word" before the space, e.g. "03" from "03 - Utils"
    first_token = category_folder.split(" ")[0]
    index_part = first_token.strip()

    if index_part in CATEGORY_INDEX_MAP:
        cat_short, cat_name = CATEGORY_INDEX_MAP[index_part]
        category_code = cat_short
        category_name = cat_name
    else:
        # Fallback: try to use the whole folder name as category_name
        category_name = category_folder
        category_code = None

    return base_model_name, base_model_code, category_name, category_code


def build_lora_record(file_path: str, root_dir: str) -> LoraRecord:
    """Create a LoraRecord for a given .safetensors file path."""
    file_path_norm = normalise_path(file_path)
    filename = os.path.basename(file_path_norm)

    base_model_name, base_model_code, category_name, category_code = parse_base_and_category(
        file_path_norm, root_dir
    )

    record = LoraRecord(
        file_path=file_path_norm,
        filename=filename,
        base_model_name=base_model_name,
        base_model_code=base_model_code,
        category_name=category_name,
        category_code=category_code,
        lora_type=None,
        rank=None,
        block_weights=None,
        clink_profile_name=None,
    )

    return record


def find_lora_files(root_dir: str) -> List[str]:
    """Recursively find all .safetensors files under the given root directory."""
    root_dir = normalise_path(root_dir)
    lora_files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".safetensors"):
                full_path = os.path.join(dirpath, name)
                lora_files.append(full_path)

    return lora_files


def main():
    print("=== LoRA Catalog Skeleton ===")
    print("This step only discovers files and parses base model + category.")
    print()

    # Default root directory (you can change this to whatever you use)
    default_root = r"E:\models\loras"

    root_dir = input(f"Enter LoRA root directory [{default_root}]: ").strip()
    if not root_dir:
        root_dir = default_root

    root_dir = normalise_path(root_dir)

    if not os.path.isdir(root_dir):
        print(f"ERROR: '{root_dir}' is not a valid directory.")
        return

    print(f"\nScanning for .safetensors under: {root_dir}\n")

    lora_paths = find_lora_files(root_dir)
    print(f"Found {len(lora_paths)} LoRA file(s).")
    print()

    # Show a small sample so you can verify parsing is correct
    max_preview = 20  # adjust if you want more/less
    for i, path in enumerate(sorted(lora_paths)):
        record = build_lora_record(path, root_dir)
        rec_dict = asdict(record)

        print(f"[{i+1}] {record.filename}")
        print(f"    Path: {record.file_path}")
        print(f"    Base model: {record.base_model_name} ({record.base_model_code})")
        print(f"    Category : {record.category_name} ({record.category_code})")
        # block_weights will be filled later
        print()

        if i + 1 >= max_preview:
            if len(lora_paths) > max_preview:
                print(f"... (showing first {max_preview} of {len(lora_paths)} files)")
            break

    print("\nDone. Once you’re happy with the base model / category parsing,")
    print("we’ll bolt on the database and Delta Inspector integration.")


if __name__ == "__main__":
    main()
