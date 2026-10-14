"""Unit tests of the real `calculate_metrics` tool function (rules: see its docstring)."""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from celltracks_lc_tools import calculate_metrics  # noqa: E402
from labconstrictor_tools import ToolError  # noqa: E402

HEADER = "Unique_ID,POSITION_T,POSITION_X,POSITION_Y,POSITION_Z\n"


def df(text):
    return pd.read_csv(io.StringIO(text))


def run(text, **kw):
    return calculate_metrics(df(text), **kw)


def test_single_spot_track_is_skipped_not_a_crash():
    table, message = run(HEADER + "A,0,0,0,0\nA,1,1,0,0\nS,0,5,5,0\n")
    assert list(table["Unique_ID"]) == ["A"]
    assert message.startswith("1 tracks, 1 skipped (fewer than 2 spots), 0 rows dropped (NaN)")


def test_only_single_spot_tracks_is_a_readable_no_result():
    with pytest.raises(ToolError) as e:
        run(HEADER + "S,0,5,5,0\n")
    assert e.value.code == "no_result"


def test_nan_rows_dropped_before_counting_spots():
    # B has 2 rows but one is NaN: it becomes a single-spot track and is skipped, as in the loader (dropna, then filter).
    table, message = run(HEADER + "A,0,0,0,0\nA,1,,0,0\nA,2,2,0,0\nB,0,0,0,0\nB,1,1,,0\n")
    assert list(table["Unique_ID"]) == ["A"]
    assert table["Total Distance Traveled"][0] == 2.0
    assert "1 skipped" in message and "2 rows dropped (NaN)" in message


def test_2d_without_z_uses_zero_and_says_so():
    table, message = run("Unique_ID,POSITION_T,POSITION_X,POSITION_Y\nA,0,0,0\nA,1,3,4\n")
    assert table["Total Distance Traveled"][0] == 5.0
    assert "Z = 0" in message


def test_empty_z_column_counts_as_2d_not_as_all_rows_dropped():
    table, message = run(HEADER + "A,0,0,0,\nA,1,3,4,\n")
    assert table["Mean Speed"][0] == 5.0 and "Z = 0" in message


def test_3d_z_is_used():
    table, message = run(HEADER + "A,0,0,0,0\nA,1,0,0,2\n")
    assert table["Total Distance Traveled"][0] == 2.0
    assert "Z = 0" not in message


def test_unsorted_time_is_ordered_per_track():
    table, _ = run(HEADER + "A,2,2,0,0\nA,0,0,0,0\nA,1,1,0,0\n")
    assert table["Track Duration"][0] == 2 and table["Mean Speed"][0] == 1.0 and table["Directionality"][0] == 1.0


def test_track_id_column_choice_and_default_unchanged():
    text = "TRACK_ID,POSITION_T,POSITION_X,POSITION_Y\n7,0,0,0\n7,1,3,4\n"
    table, _ = run(text, track_id_column="TRACK_ID")
    assert list(table.columns)[0] == "TRACK_ID"
    with pytest.raises(ToolError) as e:
        run(text)
    assert e.value.code == "missing_columns" and "track_id_column" in str(e.value)


def test_existing_output_columns_unchanged():
    table, _ = run(HEADER + "A,0,0,0,0\nA,1,1,0,0\n")
    assert list(table.columns) == [
        "Unique_ID", "Track Duration", "Mean Speed", "Median Speed", "Max Speed", "Min Speed",
        "Speed Standard Deviation", "Total Distance Traveled", "Directionality",
    ]


def test_empty_track_id_still_refused():
    with pytest.raises(ToolError) as e:
        run(HEADER + "A,0,0,0,0\n,1,1,0,0\n")
    assert e.value.code == "missing_track_id"


def test_progress_and_cancel_per_track(monkeypatch):
    import celltracks_lc_tools as mod

    calls = []
    monkeypatch.setattr(mod, "progress", lambda f, m="": calls.append(f))
    n = {"c": 0}

    def cancel():
        n["c"] += 1
        if n["c"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(mod, "check_cancel", cancel)
    with pytest.raises(KeyboardInterrupt):
        run(HEADER + "A,0,0,0,0\nA,1,1,0,0\nB,0,0,0,0\nB,1,1,0,0\nC,0,0,0,0\nC,1,1,0,0\n")
    assert calls == [1 / 3]
