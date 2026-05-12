from gemma4_mlx_mac.models import DEFAULT_MODEL_ID, list_model_profiles


def test_default_model_profile_is_present() -> None:
    profiles = list_model_profiles()

    assert any(profile.id == DEFAULT_MODEL_ID and profile.default for profile in profiles)
    assert profiles[0].recommended_memory_gb <= 16
