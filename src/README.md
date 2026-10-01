# CellTracksColab source code

This folder contains the `celltracks` Python package that all CellTracksColab notebooks import:

| Module | What it provides |
|---|---|
| `data_loader.py`, `xml_loader.py` | Loading and compiling tracking data (TrackMate CSV/XML, custom CSV, CellTracksColab format) |
| `Track_Metrics.py` | Track metric computation (speed, directionality, etc.) |
| `Track_Plots.py`, `BoxPlots_Statistics.py` | Track visualisation, box plots and statistics |
| `Dimensionality_Reduction.py` | UMAP, t-SNE and HDBSCAN helpers |
| `Track_Clustering.py` | Spatial clustering analyses (Ripley's functions) |
| `Distance_to_ROI.py` | Distance-to-ROI analyses |

How it is used:

- **Google Colab:** the first cell of each notebook clones this repository and adds `src/` to the Python path.
- **Desktop app:** the package is installed together with the app, and the Welcome notebook can update it when this folder changes.
- **From source:** run `pip install -e .` at the repository root.
