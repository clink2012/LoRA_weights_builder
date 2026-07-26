from __future__ import annotations

import lora_indexer
from model_family_integration import apply_model_family_registry


def main() -> None:
    apply_model_family_registry(lora_indexer)
    lora_indexer.main()


if __name__ == "__main__":
    main()
