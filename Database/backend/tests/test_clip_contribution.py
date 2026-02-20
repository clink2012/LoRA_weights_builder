from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from clip_contribution import is_clip_contributor  # noqa: E402


def test_is_clip_contributor_true_when_te1_key_present():
    keys = [
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight",
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight",
    ]

    is_contributor, count = is_clip_contributor(keys)

    assert is_contributor is True
    assert count == 1


def test_is_clip_contributor_false_for_unet_only_keys():
    keys = [
        "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight",
        "lora_unet_down_blocks_1_attentions_1_to_k.lora_up.weight",
    ]

    is_contributor, count = is_clip_contributor(keys)

    assert is_contributor is False
    assert count == 0
