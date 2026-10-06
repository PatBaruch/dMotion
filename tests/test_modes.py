from dataclasses import replace

import pytest

from dmotion.config import AppConfig, apply_mode


@pytest.mark.parametrize("mode", ["money", "trained"])
def test_cash_aliases_preserve_selected_checkpoint_and_confidence(tmp_path, mode):
    config = AppConfig(root=tmp_path)
    config = replace(
        config, detector=replace(config.detector, model="candidate.pt", confidence=0.73)
    )
    assert apply_mode(config, mode) is config


@pytest.mark.parametrize("mode", ["check", "reference", "unknown"])
def test_retired_modes_fail_explicitly(tmp_path, mode):
    with pytest.raises(ValueError, match="Retired detection mode"):
        apply_mode(AppConfig(root=tmp_path), mode)
