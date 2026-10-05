# Tests for the tools exposed to Napari / Fiji

Declarations: `src/celltracks_lc_tools/__init__.py`. They run in CellTracksColab's own environment through the LabConstrictor tools worker.

    labconstrictor-tools check --module celltracks_lc_tools
    labconstrictor-tools test  --module celltracks_lc_tools --cases lc_tests/cases.json

Fixtures are tiny (two tracks). The cases check the computed columns and that a CSV with missing columns fails with a readable error.
