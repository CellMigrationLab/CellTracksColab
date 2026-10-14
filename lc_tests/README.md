# Tests for the tools exposed to Napari / Fiji

Declarations: `src/celltracks_lc_tools/__init__.py`. They run in CellTracksColab's own environment through the LabConstrictor tools worker.

    labconstrictor-tools check --module celltracks_lc_tools
    labconstrictor-tools test  --module celltracks_lc_tools --cases lc_tests/cases.json

Fixtures are tiny. Cases cover the computed columns, missing columns, empty ids, single-spot tracks (skipped), empty (NaN) positions (rows dropped), 2D data without POSITION_Z, unsorted time and a raw TRACK_ID column. Unit tests: `tests/test_calculate_metrics.py`.
