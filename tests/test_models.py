from gemma4_mlx_mac.models import DEFAULT_MODEL_ID, get_model_profile, list_model_profiles


def test_default_model_profile_is_qat_4bit() -> None:
    profiles = list_model_profiles()

    assert DEFAULT_MODEL_ID == "mlx-community/gemma-4-E2B-it-qat-4bit"
    assert any(profile.id == DEFAULT_MODEL_ID and profile.default for profile in profiles)
    assert profiles[0].qat is True
    assert profiles[0].recommended_memory_gb <= 16


def test_12b_and_26b_qat_profiles_are_present() -> None:
    profile_12b = get_model_profile("mlx-community/gemma-4-12B-it-qat-4bit")
    profile_26b = get_model_profile("mlx-community/gemma-4-26B-A4B-it-qat-4bit")

    assert profile_12b is not None
    assert profile_12b.modality == "any-to-any"
    assert profile_12b.context_length == 256_000
    assert profile_12b.qat is True

    assert profile_26b is not None
    assert profile_26b.recommended_memory_gb <= 32
    assert profile_26b.qat is True


def test_bf16_profiles_use_canonical_lowercase_repo_ids() -> None:
    assert get_model_profile("mlx-community/gemma-4-26b-a4b-it-bf16") is not None
    assert get_model_profile("mlx-community/gemma-4-31b-it-bf16") is not None
