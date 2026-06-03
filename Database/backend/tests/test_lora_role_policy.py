from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_role_policy import get_role_policy, list_role_policies  # noqa: E402


def test_role_policy_returns_character_as_identity_anchor():
    policy = get_role_policy("character")

    assert policy.role == "character"
    assert policy.priority == 100
    assert policy.default_model_strength == 0.90
    assert policy.default_clip_strength == 0.70
    assert policy.intent_label == "identity anchor"
    assert policy.protects_identity is True


def test_role_policy_canonicalizes_action_and_pose_to_utility():
    action = get_role_policy("action")
    pose = get_role_policy("pose")

    assert action.role == "utility"
    assert pose.role == "utility"
    assert action.default_model_strength == 0.35
    assert pose.default_clip_strength == 0.00
    assert action.preserves_composition is True


def test_role_policy_defaults_unknown_roles_to_other():
    policy = get_role_policy("weird-folder-name")

    assert policy.role == "other"
    assert policy.priority == 20
    assert policy.intent_label == "unknown intent"


def test_role_policies_follow_existing_role_hierarchy_order():
    policies = list_role_policies()

    assert [policy.role for policy in policies] == [
        "character",
        "style",
        "clothing",
        "environment",
        "utility",
        "other",
    ]


def test_role_policy_priorities_encode_non_equal_artist_intent():
    character = get_role_policy("character")
    clothing = get_role_policy("clothing")
    style = get_role_policy("style")
    utility = get_role_policy("utility")

    assert character.priority > clothing.priority > style.priority > utility.priority
    assert character.protects_identity is True
    assert style.treat_as_flavour is True
