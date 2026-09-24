"""Legacy split assertion retained to keep the audited old artifact reproducible."""

from src.data.splits import FINE_TUNE_GROUP_COLUMNS


def test_legacy_group_key_matches_the_recorded_old_artifact() -> None:
    assert FINE_TUNE_GROUP_COLUMNS == [
        "output_power_w",
        "phase_shift_deg",
        "dBdt_max",
        "B_thd_percent",
        "form_factor",
    ]
