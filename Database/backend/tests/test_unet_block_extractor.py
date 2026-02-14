from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from unet_block_extractor import extract_unet_57_block_weights


def _write_mapped_fixture(path: Path) -> None:
    tensors = {
        "lora_unet_conv_in.lora_down.weight": torch.ones((2, 2)),
        "lora_unet_time_embedding_linear_1.lora_down.weight": torch.ones((2, 2)) * 2,
        "lora_unet_time_embedding_linear_2.lora_down.weight": torch.ones((2, 2)) * 3,
        "lora_unet_add_embedding.lora_down.weight": torch.ones((2, 2)) * 4,
        "lora_unet_conv_norm_out.lora_down.weight": torch.ones((2, 2)) * 55,
        "lora_unet_conv_out.lora_down.weight": torch.ones((2, 2)) * 56,
    }

    for i in range(4):
        tensors[f"lora_unet_down_blocks_{i}_resnets_0_conv1.lora_down.weight"] = torch.ones((2, 2)) * (5 + i)
        tensors[f"lora_unet_down_blocks_{i}_resnets_1_conv1.lora_down.weight"] = torch.ones((2, 2)) * (6 + i)
        tensors[f"lora_unet_down_blocks_{i}_attentions_0_to_q.lora_down.weight"] = torch.ones((2, 2)) * (7 + i)
        tensors[f"lora_unet_down_blocks_{i}_attentions_1_to_q.lora_down.weight"] = torch.ones((2, 2)) * (8 + i)
        tensors[f"lora_unet_down_blocks_{i}_downsamplers_0_conv.lora_down.weight"] = torch.ones((2, 2)) * (9 + i)

        tensors[f"lora_unet_up_blocks_{i}_resnets_0_conv1.lora_down.weight"] = torch.ones((2, 2)) * (30 + i)
        tensors[f"lora_unet_up_blocks_{i}_resnets_1_conv1.lora_down.weight"] = torch.ones((2, 2)) * (31 + i)
        tensors[f"lora_unet_up_blocks_{i}_resnets_2_conv1.lora_down.weight"] = torch.ones((2, 2)) * (32 + i)
        tensors[f"lora_unet_up_blocks_{i}_attentions_0_to_q.lora_down.weight"] = torch.ones((2, 2)) * (33 + i)
        tensors[f"lora_unet_up_blocks_{i}_attentions_1_to_q.lora_down.weight"] = torch.ones((2, 2)) * (34 + i)
        tensors[f"lora_unet_up_blocks_{i}_attentions_2_to_q.lora_down.weight"] = torch.ones((2, 2)) * (35 + i)
        tensors[f"lora_unet_up_blocks_{i}_upsamplers_0_conv.lora_down.weight"] = torch.ones((2, 2)) * (36 + i)

    tensors["lora_unet_mid_block_resnets_0_conv1.lora_down.weight"] = torch.ones((2, 2)) * 40
    tensors["lora_unet_mid_block_attentions_0_to_q.lora_down.weight"] = torch.ones((2, 2)) * 41
    tensors["lora_unet_mid_block_resnets_1_conv1.lora_down.weight"] = torch.ones((2, 2)) * 42

    save_file(tensors, str(path))


def test_mapping_sanity_57(tmp_path: Path) -> None:
    fixture = tmp_path / "mapped.safetensors"
    _write_mapped_fixture(fixture)

    weights = extract_unet_57_block_weights(str(fixture))

    assert len(weights) == 57
    assert set(range(len(weights))) == set(range(57))


def test_determinism(tmp_path: Path) -> None:
    fixture = tmp_path / "mapped.safetensors"
    _write_mapped_fixture(fixture)

    first = extract_unet_57_block_weights(str(fixture))
    second = extract_unet_57_block_weights(str(fixture))

    assert first == second


def test_unknown_unet_keys_fail(tmp_path: Path) -> None:
    fixture = tmp_path / "unknown.safetensors"
    save_file(
        {"lora_unet_weird_branch_99_tensor.lora_down.weight": torch.ones((2, 2))},
        str(fixture),
    )

    with pytest.raises(ValueError, match="could not be mapped"):
        extract_unet_57_block_weights(str(fixture))
