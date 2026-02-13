import os
from typing import Dict

import torch
from safetensors import safe_open


def normalise_path(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def list_keys(path: str) -> Dict[str, str]:
    """
    Return a dict of {key_name: str(dtype_shape)} for a safetensors file.
    Uses Torch backend so it can handle bfloat16.
    """
    path = normalise_path(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    info: Dict[str, str] = {}

    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            info[key] = f"{str(t.dtype)} {tuple(t.shape)}"

    return info


def main():
    print("=== Safetensors Key Lister ===")
    print("This helper prints all tensor keys in a .safetensors file.")
    print()

    file_path = input("LoRA .safetensors path: ").strip().strip('"')

    if not file_path:
        print("No path provided, aborting.")
        return

    file_path = normalise_path(file_path)

    try:
        info = list_keys(file_path)
    except Exception as e:
        print("\nERROR while reading file:")
        print(str(e))
        return

    print(f"\nFound {len(info)} tensor(s) in:\n  {file_path}\n")
    print("First 50 keys (or fewer if file has less):")
    print("------------------------------------------")

    for i, (k, v) in enumerate(info.items()):
        print(f"[{i+1:3}] {k} -> {v}")
        if i + 1 >= 50:
            if len(info) > 50:
                print(f"... ({len(info) - 50} more keys not shown)")
            break


if __name__ == "__main__":
    main()
