<table>
<tr>
<td valign="top">

<img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/CellTracksColab_logo.png" width="800">

</td>
<td>

> In life sciences, tracking objects from movies is pivotal for quantifying behaviors of particles, organelles, bacteria, cells, and whole animals. **CellTracksColab** bridges the gap between tracking and analysis.

> **CellTracksColab** simplifies the journey from data compilation to analysis.

</td>
</tr>
</table>

---

## 🚀 **Key Features**
- 📘 **Holistic View**: Comprehensive analysis across fields of view, biological repeats, and conditions.
- 🖥️ **User-Centric**: Intuitive GUI designed for all users.
- 🔍 **Visualization**: Track visualization and filtering.
- 📊 **Analysis**: Deep-dive into track metrics and statistics.
- 🧪 **Reliability**: Check experimental variability using hierarchical clustering.
- 🔧 **Advanced Tools**: Harness the power of UMAP, t-SNE, and HDBSCAN.
- 💼 **Flexibility**: Tailor and adapt to your needs.

---

## ✅ **Compatible with**
- [TrackMate](http://imagej.net/TrackMate)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [Icy](http://icy.bioimageanalysis.org/)
- [ilastik](https://www.ilastik.org/)
- [Fiji Manual Tracker](https://imagej.net/Fiji)

## 🛠️ **Quick Start**

The easiest way to start using **CellTracksColab** is in the cloud using Google Collaboratory, but it can also be used on your own computer using Jupyter Notebooks. See our [wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki) for installation instructions.

### 1. **Load and Plot Your Data**

We provide three notebooks for loading and analyzing your data depending on its format:

#### For TrackMate Data (CSV or XML files):
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate.ipynb)
- See how to prepare the data in the [TrackMate notebook wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki/The-TrackMate-notebook).

#### For Data Generated in CellProfiler, ICY, ilastik, or the Fiji Manual Tracker (CSV files):
- Explore our general-purpose notebook for analyzing diverse tracking datasets.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Custom.ipynb)
- See how to prepare the data in the [General notebook wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki/The-General-notebook).

#### For Data Already in the CellTracksColab Format:
- **CellTracksColab - Viewer**: Ideal for loading datasets in CellTracksColab format or sharing data with colleagues.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Viewer.ipynb)

### 2. **Advanced Analysis Modules**

These notebooks require your dataset to be in the CellTracksColab format.

#### CellTracksColab - Dimensionality Reduction: 
  - Utilize advanced dimensionality reduction techniques to facilitate the interpretation of complex, high-dimensional data.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Dimensionality_Reduction.ipynb)

#### CellTracksColab - Track Spatial Clustering Analysis:
  - Dive deeper into your dataset with our track clustering analysis module.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Track_Clustering.ipynb)

#### CellTracksColab - Distance to ROI: 
  - Analyze movement tracks in relation to designated Regions of Interest (ROIs). Compute and analyze the distances between moving objects (tracks) and dynamic ROIs.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Distance_to_ROI.ipynb)

#### More to come

##  Other Notebooks

#### CellTracksColab - TrackMate - Plate:
  - Handle TrackMate data structured in a plate format, such as file names commonly produced by incubator microscopes like Incucytes.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate_Plate.ipynb)
    
---

## ⭐️ **Acknowledgments**

CellTracksColab is inspired by several key projects in cell tracking and analysis. We acknowledge the influential contributions of **[Traject3d](https://www.nature.com/articles/s41467-022-32958-x)**, **[CellPhe](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10070448/)**, **[CelltrackR](https://www.sciencedirect.com/science/article/pii/S2667119021000033)**, the **[MotilityLab website](https://www.motilitylab.net/)**, and **[Cellplato on Zenodo](https://zenodo.org/records/8096717)**. The innovative use of UMAP and HDBSCAN for analyzing tracking data, as featured in CellTracksColab, was first introduced in **[CellPlato](https://github.com/Michael-shannon/cellPLATO)**.

---
## 📦 **Resources**
- **Test Dataset**: Start exploring with our test datasets in [CellTracksColab CSV format](https://zenodo.org/records/8420011), or [TrackMate CSV format](https://zenodo.org/records/8413510).
- **Data Structure**: Organize with our two-tiered folder hierarchy. [Details here](https://github.com/CellMigrationLab/CellTracksColab/wiki/Prepare-the-Session-and-Load-Your-Data#3-navigating-to-your-dataset-on-google-drive).
- **Data Requirements**: Note that **CellTracksColab** does not yet support track merging or splitting.

---

## 📚 **Documentation**
Dive deeper. [Visit our comprehensive wiki](https://github.com/guijacquemet/CellTracksColab/wiki).

---

## ✍️ **Contributors**

- [Estibaliz Gómez-de-Mariscal](https://esgomezm.github.io/)
- [Hanna Grobe](https://www.abo.fi/en/contact/hanna-grobe/)
- [Joanna W. Pylvänäinen](https://research.abo.fi/en/persons/joanna-pylv%C3%A4n%C3%A4inen)
- [Laura Xénard](https://research.pasteur.fr/en/member/laura-xenard/)
- [Ricardo Henriques](https://henriqueslab.github.io/)
- [Jean-Yves Tinevez](https://research.pasteur.fr/en/member/jean-yves-tinevez/)
- [Guillaume Jacquemet](https://cellmig.org/)

---

## 🤝 **Contribute**
We welcome your insights and improvements. Raise an [issue here](https://github.com/guijacquemet/CellTracksColab/issues).

---

## **License**
Licensed under the MIT License. [Details here](https://opensource.org/licenses/MIT).

---

## 📜 **Citation**

If you use CellTracksColab in your research, please cite the following paper:

### Reference
Guillaume Jacquemet. (2023). CellTracksColab—A platform for compiling, analyzing, and exploring tracking data. *bioRxiv*. [https://doi.org/10.1101/2023.10.20.563252](https://doi.org/10.1101/2023.10.20.563252)

[Download PDF](https://www.biorxiv.org/content/early/2023/10/26/2023.10.20.563252.full.pdf)


## 🖼️ **Screenshots**

<table>
<tr>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot1.png" alt="Screenshot 1" width="400"/>
        <br>
        <em>Figure 1: Compile your data</em>
    </td>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot2.png" alt="Screenshot 2" width="400"/>
        <br>
        <em>Figure 2: Visualise your tracks</em>
    </td>
</tr>
<tr>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot3.png" alt="Screenshot 3" width="400"/>
        <br>
        <em>Figure 3: Compute additional metrics</em>
    </td>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot4.png" alt="Screenshot 4" width="400"/>
        <br>
        <em>Figure 4: Plot track parameters</em>
    </td>
</tr>
<tr>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot5.png" alt="Screenshot 5" width="400"/>
        <br>
        <em>Figure 5: Compute Similarity Metrics between Field of Views and between Conditions and Repeats</em>
    </td>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot6.png" alt="Screenshot 6" width="400"/>
        <br>
        <em>Figure 6: Perform UMAP</em>
    </td>
</tr>
<tr>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot7.png" alt="Screenshot 7" width="400"/>
        <br>
        <em>Figure 7: Identify clusters using HDBSCAN</em>
    </td>
    <td align="center" valign="middle">
        <img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/Screenshot8.png" alt="Screenshot 8" width="400"/>
        <br>
        <em>Figure 8: Understand your clusters using a heatmap</em>
    </td>
</tr>
</table>


---



