"""CellTracksColab tools for Napari, Fiji and the command line (LabConstrictor tools bridge, experimental).

This module only *declares* the tools: `labconstrictor_tools` and the standard library are imported at the top, CellTracksColab and
pandas inside the function, so that listing the tools stays instant. It lives in its own package (not inside `celltracks`) because
importing `celltracks` pulls in pandas, requests and notebook widgets.

    labconstrictor-tools check --module celltracks_lc_tools
    labconstrictor-tools test  --module celltracks_lc_tools --cases lc_tests/cases.json
"""

from typing import Annotated

from labconstrictor_tools import Description, Table, TableOut, ToolError, tool


@tool("Calculate Track Metrics")
def calculate_metrics(
    tracks: Annotated[Table, Description("CSV with columns Unique_ID, POSITION_T, POSITION_X, POSITION_Y, POSITION_Z")],
) -> TableOut:
    """Duration, speed statistics, distance and directionality per track (Unique_ID)."""
    from celltracks.Track_Metrics import calculate_directionality, calculate_track_metrics

    missing = [
        c
        for c in ("Unique_ID", "POSITION_T", "POSITION_X", "POSITION_Y", "POSITION_Z")
        if c not in tracks.columns
    ]
    if missing:
        raise ToolError("missing_columns", "CSV is missing columns: " + ", ".join(missing))
    unlabelled = tracks["Unique_ID"].isna()
    if unlabelled.any():  # groupby would drop these rows without a word: their tracks would be missing from the result
        raise ToolError(
            "missing_track_id",
            "%d row(s) have an empty Unique_ID (first data rows: %s). Give every row a track id or remove those rows."
            % (int(unlabelled.sum()), ", ".join(str(i + 1) for i in tracks.index[unlabelled][:5])),
        )
    grouped = tracks.groupby("Unique_ID")
    return grouped.apply(calculate_track_metrics).join(grouped.apply(calculate_directionality)).reset_index()
