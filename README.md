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
<table>
  <tr>
    <th><a href="http://imagej.net/TrackMate"><img src="https://imagej.net/media/icons/trackmate.png" alt="TrackMate Logo" width="100"></a></th>
    <th><a href="https://github.com/CellProfiler/CellProfiler"><img src="https://avatars.githubusercontent.com/u/710590?s=280&v=4" alt="CellProfiler Logo" width="100"></a></th>
    <th><a href="http://icy.bioimageanalysis.org/"><img src="https://icy.bioimageanalysis.org/wp-content/uploads/2018/07/logo_full_notext600px.png" alt="Icy Logo" width="100"></a></th>
    <th><a href="https://www.ilastik.org/"><img src="https://www.ilastik.org/assets/ilastik-logo.png" alt="ilastik Logo" width="100"></a></th>
    <th><a href="https://imagej.net/Fiji"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/FIJI_%28software%29_Logo.svg/1200px-FIJI_%28software%29_Logo.svg.png" alt="Fiji Logo" width="100"></a></th>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="http://imagej.net/TrackMate">TrackMate</a></td>
    <td style="text-align: center;"><a href="https://github.com/CellProfiler/CellProfiler">CellProfiler</a></td>
    <td style="text-align: center;"><a href="http://icy.bioimageanalysis.org/">Icy</a></td>
    <td style="text-align: center;"><a href="https://www.ilastik.org/">ilastik</a></td>
    <td style="text-align: center;"><a href="https://imagej.net/Fiji">Fiji Manual Tracker</a></td>
  </tr>
</table>

But may also be compatible with other tracking software exporting tracking results that meet our minimal requirements. Ensure your data is well-organized according to the recommended folder hierarchy. This structure helps in managing and analyzing the data efficiently.

- [Check the data requirements and the supported software](https://github.com/CellMigrationLab/CellTracksColab/wiki/Data-requirements-and-supported-software)

## 🛠️ **Quick Start**

The easiest way to start using **CellTracksColab** is in the cloud using Google Collaboratory, but it can also be used on your own computer using Jupyter Notebooks. See our [wiki](https://github.com/CellMigrationLab/CellTracksColab/wiki) for installation instructions.

### 1. **Load and Plot Your Data**
We provide three notebooks for loading and analyzing your data depending on its format:

<table>
  <tr>
    <th>Notebook</th>
    <th>Purpose</th>
    <th>Required File Format</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>CellTracksColab - TrackMate</strong></td>
    <td>Load and analyze TrackMate data. More info <a href="https://github.com/CellMigrationLab/CellTracksColab/wiki/The-TrackMate-notebook">here</a>.</td>
    <td>CSV or XML files</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
  <tr>
    <td><strong>CellTracksColab - Custom</strong></td>
    <td>Analyze data from CellProfiler, ICY, ilastik, or Fiji Manual Tracker. More info <a href="https://github.com/CellMigrationLab/CellTracksColab/wiki/The-Custom-notebook">here</a>.</td>
    <td>CSV files</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Custom.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
  <tr>
    <td><strong>CellTracksColab - Viewer</strong></td>
    <td>Load and share data in the CellTracksColab format.</td>
    <td>CellTracksColab format</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Viewer.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
</table>

### 2. **Advanced Analysis Modules**

These notebooks require your dataset to be in the CellTracksColab format.

<table>
  <tr>
    <th>Notebook</th>
    <th>Purpose</th>
    <th>Required File Format</th>
    <th>Link</th>
  </tr>
  <tr>
    <td><strong>CellTracksColab - Dimensionality Reduction</strong></td>
    <td>Utilize advanced dimensionality reduction techniques.</td>
    <td>CellTracksColab format</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Dimensionality_Reduction.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
  <tr>
    <td><strong>CellTracksColab - Track Spatial Clustering Analysis</strong></td>
    <td>Dive deeper into your dataset with track clustering analysis.</td>
    <td>CellTracksColab format</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Track_Clustering.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
  <tr>
    <td><strong>CellTracksColab - Distance to ROI</strong></td>
    <td>Analyze movement tracks in relation to designated ROIs.</td>
    <td>CellTracksColab format</td>
    <td>
      <a href="https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_Distance_to_ROI.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
      </a>
    </td>
  </tr>
</table>

More to come

##  Other Notebooks

#### CellTracksColab - TrackMate - Plate:
  - Handle TrackMate CSV files structured in a plate format, such as file names commonly produced by incubator microscopes like Incucytes.
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

We welcome your insights and improvements! There are several ways you can contribute to the CellTracksColab project:

### Issues
If you encounter any bugs, have suggestions for improvements, or want to discuss new features, please raise an issue on our [GitHub Issues page](https://github.com/CellMigrationLab/CellTracksColab/issues).

### New Analysis Notebooks
We are excited to see new analysis notebooks built on the CellTracksColab platform. If you have developed a new notebook, please submit it via a pull request. All submitted notebooks should include a test dataset to showcase their functionality. Each notebook will be tested by a member of the team before being released.

### Code of Conduct
We expect all contributors to adhere to our simple code of conduct:

- Be respectful and considerate of others.
- Provide constructive feedback.
- Collaborate openly and honestly.

By participating in this project, you agree to abide by these guidelines.

---

Thank you for contributing to CellTracksColab! Your support and contributions help us improve and expand the platform for everyone in the community.

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



