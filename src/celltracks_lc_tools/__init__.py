"""CellTracksColab tools for Napari, Fiji and the command line (LabConstrictor tools bridge, experimental).

This module only *declares* the tools: `labconstrictor_tools` and the standard library are imported at the top, CellTracksColab and
pandas inside the function, so that listing the tools stays instant. It lives in its own package (not inside `celltracks`) because
importing `celltracks` pulls in pandas, requests and notebook widgets.

    labconstrictor-tools check --module celltracks_lc_tools
    labconstrictor-tools test  --module celltracks_lc_tools --cases lc_tests/cases.json
"""

from typing import Annotated

from labconstrictor_tools import (
    Description,
    MessageOut,
    Name,
    Table,
    TableOut,
    ToolError,
    check_cancel,
    progress,
    tool,
)

_COORDS = ("POSITION_T", "POSITION_X", "POSITION_Y")


@tool("Calculate Track Metrics")
def calculate_metrics(
    tracks: Annotated[
        Table,
        Description(
            "CSV with one row per spot: a track id column (default Unique_ID), POSITION_T, POSITION_X, POSITION_Y and, "
            "for 3D data, POSITION_Z. Without POSITION_Z the data is treated as 2D (Z = 0). Positions and times in the "
            "units of your file: speeds are in those position units per time unit."
        ),
    ],
    track_id_column: Annotated[
        str, Description("Name of the column that identifies the track of each spot, e.g. TRACK_ID. Default: Unique_ID")
    ] = "Unique_ID",
) -> tuple[Annotated[TableOut, Name("table")], Annotated[MessageOut, Name("summary")]]:
    """Duration, speed statistics, distance and directionality per track.

    Rules, the same as the CellTracksColab notebooks and loader: rows with an empty POSITION_T/X/Y/Z are dropped
    first (empty POSITION_Z only when the column exists and is not empty everywhere), then tracks with fewer than 2
    spots are skipped (a speed needs two spots). An infinite position or time, or two spots of one track at the same time, is refused
    with a sentence (the speeds would be infinite or absurd). Spots are ordered by POSITION_T inside each track. A file without
    POSITION_Z (2D) is processed with Z = 0. The summary says how many tracks, skipped tracks and dropped rows there were.

    Args:
        tracks: The spots table.
        track_id_column: Column holding the track id of each spot.
    """
    import numpy as np
    import pandas as pd
    from celltracks.Track_Metrics import calculate_directionality, calculate_track_metrics

    id_col = track_id_column.strip() if track_id_column else "Unique_ID"
    missing = [c for c in (id_col,) + _COORDS if c not in tracks.columns]
    if missing:
        hint = ""
        if id_col not in tracks.columns:
            others = [c for c in ("Unique_ID", "TRACK_ID") if c in tracks.columns]
            if others:
                hint = " (set 'track_id_column' to %s)" % " or ".join(others)
        raise ToolError("missing_columns", "CSV is missing columns: " + ", ".join(missing) + hint)
    unlabelled = tracks[id_col].isna()
    if unlabelled.any():  # groupby would drop these rows without a word: their tracks would be missing from the result
        raise ToolError(
            "missing_track_id",
            "%d row(s) have an empty %s (first data rows: %s). Give every row a track id or remove those rows."
            % (int(unlabelled.sum()), id_col, ", ".join(str(i + 1) for i in tracks.index[unlabelled][:5])),
        )

    coords = list(_COORDS)
    data = tracks[[id_col] + coords].copy()
    has_z = "POSITION_Z" in tracks.columns and not tracks["POSITION_Z"].isna().all()
    if has_z:
        data["POSITION_Z"] = tracks["POSITION_Z"]
        coords.append("POSITION_Z")
    else:
        data["POSITION_Z"] = 0.0
    for c in coords:  # text that is not a number counts as empty
        data[c] = pd.to_numeric(data[c], errors="coerce")
    infinite = ~np.isfinite(data[coords].to_numpy(dtype=float)) & ~data[coords].isna().to_numpy()  # NaN is an empty value (dropped below); infinity is a wrong value
    if infinite.any():
        first = ", ".join(str(i + 1) for i in data.index[infinite.any(axis=1)][:5])
        raise ToolError(
            "non_finite_position",
            "%d row(s) have an infinite position or time (first data rows: %s): the speeds would be infinite. Correct or remove those rows."
            % (int(infinite.any(axis=1).sum()), first),
        )
    before = len(data)
    data = data.dropna(subset=coords)
    dropped = before - len(data)

    sizes = data.groupby(id_col).size()
    skipped = int((sizes < 2).sum())
    data = data[data[id_col].isin(sizes.index[sizes >= 2])]
    n_tracks = int(data[id_col].nunique())
    if n_tracks == 0:
        raise ToolError(
            "no_result",
            "No track has 2 or more spots with complete positions (%d track(s) skipped, %d row(s) dropped because of empty values)."
            % (skipped, dropped),
        )

    twice = data.duplicated(subset=[id_col, "POSITION_T"], keep=False)
    if twice.any():  # the library would divide by a time difference of 0 (replaced by 1e-10): an absurd speed, not a measurement
        clash = data[twice].iloc[0]
        raise ToolError(
            "same_time_twice",
            "%d spot(s) share a time with another spot of the same track (for example track %s at time %s): a speed needs different times. "
            "Remove the duplicate spots; tracks that split or merge are not supported by these statistics."
            % (int(twice.sum()), clash[id_col], clash["POSITION_T"]),
        )

    rows = []
    for k, (track_id, group) in enumerate(data.groupby(id_col), start=1):
        check_cancel()
        row = calculate_track_metrics(group).to_dict()
        row.update(calculate_directionality(group).to_dict())
        rows.append({id_col: track_id, **row})
        progress(k / n_tracks, "track %d of %d" % (k, n_tracks))
    result = pd.DataFrame(rows)

    message = "%d tracks, %d skipped (fewer than 2 spots), %d rows dropped (NaN)" % (n_tracks, skipped, dropped)
    if not has_z:
        message += ". No POSITION_Z: treated as 2D (Z = 0)"
    return result, message
