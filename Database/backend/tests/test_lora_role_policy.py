from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lora_role_policy import (  # noqa: E402
    build_role_recommendation_notes,
    build_role_strength_recommendation,
    get_role_policy,
    list_role_policies,
)


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


def test_role_recommendation_notes_are_advisory_and_role_specific():
    character_notes = build_role_recommendation_notes("character")
    style_notes = build_role_recommendation_notes("style")
    utility_notes = build_role_recommendation_notes("utility")
    other_notes = build_role_recommendation_notes("not-a-known-role")

    assert any("advisory only" in note for note in character_notes)
    assert any("identity anchor detected" in note for note in character_notes)
    assert any("flavour layer detected" in note for note in style_notes)
    assert any("composition-preserving helper detected" in note for note in utility_notes)
    assert any("unknown role detected" in note for note in other_notes)


def test_role_strength_recommendation_is_advisory_only_for_character():
    recommendation = build_role_strength_recommendation(
        "character",
        requested_model_strength=1.25,
        overlap_corrected_model_strength=0.42,
        requested_clip_strength=0.33,
        clip_contributor=True,
    )

    assert recommendation.role == "character"
    assert recommendation.requested_model_strength == 1.25
    assert recommendation.overlap_corrected_model_strength == 0.42
    assert recommendation.role_default_model_strength == 0.90
    assert recommendation.recommended_model_strength == 0.90
    assert recommendation.requested_clip_strength == 0.33
    assert recommendation.role_default_clip_strength == 0.70
    assert recommendation.recommended_clip_strength == 0.70
    assert recommendation.clip_contributor is True
    assert recommendation.applied_to_math is False
    assert recommendation.basis == "role_policy_advisory"


def test_role_strength_recommendation_disables_clip_for_non_clip_contributors():
    recommendation = build_role_strength_recommendation(
        "style",
        requested_model_strength=1.0,
        overlap_corrected_model_strength=0.25,
        requested_clip_strength=0.99,
        clip_contributor=False,
    )

    assert recommendation.role == "style"
    assert recommendation.role_default_clip_strength == 0.40
    assert recommendation.recommended_clip_strength == 0.0
    assert recommendation.clip_contributor is False
    assert recommendation.applied_to_math is False


def test_role_strength_recommendation_payload_is_stable_and_explicit():
    payload = build_role_strength_recommendation(
        "pose",
        requested_model_strength=0.8,
        overlap_corrected_model_strength=0.2,
        requested_clip_strength=0.5,
        clip_contributor=True,
    ).to_payload()

    assert payload == {
        "role": "utility",
        "requested_model_strength": 0.8,
        "overlap_corrected_model_strength": 0.2,
        "role_default_model_strength": 0.35,
        "recommended_model_strength": 0.35,
        "requested_clip_strength": 0.5,
        "role_default_clip_strength": 0.0,
        "recommended_clip_strength": 0.0,
        "clip_contributor": True,
        "applied_to_math": False,
        "basis": "role_policy_advisory",
    }
