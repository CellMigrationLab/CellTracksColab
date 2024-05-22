
<table>
<tr>
<td valign="top">

<img src="https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/CellTracksColab_logo.png" width="800">

</td>
<td>

> In life sciences, tracking objects from movies is pivotal for quantifying behaviors of particles, organelles, bacteria, cells, and whole animals. **CellTracksColab** bridges the gap between tracking and insightful analysis.

> CellTracksColab simplifies the journey from data compilation to analysis. Built on the Google Colaboratory framework, it provides a cloud-based solution accessible with just a web browser and a Google account.

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
- 🔧 **Advanced Tools**: Harness the power of UMAP, t-SNE and HDBSCAN.
- 💼 **Flexibility**: Tailor and adapt to your needs.

---

## 🛠️ **Quick Start**

To begin your analysis journey, click the "Open In Colab" button below, corresponding to your data type. For a seamless experience, right-click the button and select "Open in a new tab."

1. **For TrackMate Data**:
   - Delve into your TrackMate data with our specialized notebook.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate.ipynb)
   - See how prepare the data in the [TrackMate notebook wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki/The-TrackMate-notebook).

2. **Analysis for Other Data Types**:
   - Explore our general-purpose notebook for analyzing diverse tracking datasets.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Custom.ipynb)
   - currently datasets (.csv) generated using CellProfiler, ICY, ilastik, and Fiji Manual tracker have been succesfully tested.
   - See how prepare the data in the [General notebook wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki/The-General-notebook).
---

## 🗒️ **Additional Notebooks**

Maximize your analysis with our supplementary tools and modules.

### 🌐 **Data Viewing and Sharing**

- **CellTracksColab - Viewer**:
  - Ideal for loading datasets in CellTracksColab format or sharing data with colleagues.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Viewer.ipynb)

### 🔍 **Alternative data loading strategy**
- **CellTracksColab - TrackMate - Plate**:
  - This notebook can handle TrackMate data structured in a plate format, such as file names commonly produced by incubator microscopes like Incucytes.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate_Plate.ipynb)

### 📈 **Advanced Analysis Modules**

- **CellTracksColab - Track Clustering Analysis**:
  - Dive deeper into your dataset with our track clustering analysis module.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Track_Clustering.ipynb)

- **CellTracksColab - Distance to ROI**: 
   - This notebook is specifically designed to analyze movement tracks in relation to designated Regions of Interest (ROIs). Its primary aim is to compute and analyze the distances between moving objects (tracks) and ROIs, which may also be dynamic.
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Distance_to_ROI.ipynb)

- **CellTracksColab - Dimensionality Reduction**: 
   - This notebook is designed for analyzing datasets stored in the CellTracksColab format, utilizing advanced dimensionality reduction techniques to facilitate the interpretation of complex, high-dimensional data. To convert the data in CellTrackColab format, please use the corresponding [Quick Start notebooks](https://github.com/CellMigrationLab/CellTracksColab/tree/main?tab=readme-ov-file#%EF%B8%8F-quick-start).
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Dimensionality_Reduction.ipynb)


- More to come soon

  
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

## ✍️ **Contributor(s)**
Created by [Guillaume Jacquemet](https://cellmig.org/).

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

---



