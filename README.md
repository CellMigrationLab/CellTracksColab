# 🌟 CellTracksColab 🌟

![CellTracksColab Logo]([path_to_logo_image](https://github.com/guijacquemet/CellTracksColab/blob/main/Wiki/CellTracksColab_logo.png))

> In life sciences, tracking objects from movies is pivotal in quantifying the behavior of single particles, organelles, bacteria, cells, and even whole animals. While numerous tools allow automated tracking from video, a significant challenge remains in compiling, analyzing, and exploring the vast datasets generated. **CellTracksColab** is here to bridge this gap.

---

## 🎯 Purpose
**CellTracksColab** simplifies compiling, analyzing, and exploring tracking data. **CellTracksColab** operates on the **Google Colaboratory** framework. With its cloud-based nature, all you need is a web browser and a Google account.

---

## 🚀 Key Features
- 📘 **Holistic Dataset Overview**: Combine and analyze tracking results across multiple fields.
- 🖥️ **User-Friendly Interface**: Intuitive GUI for all users.
- 🔍 **Visualization and Filtering**: Visualize tracks and filter as needed.
- 📊 **In-depth Analysis**: Get detailed statistics for track metrics.
- 🧪 **Assess Variability**: Check experimental variability with hierarchical clustering.
- 🔧 **Powerful Data Analysis**: Use UMAP and HDBSCAN for advanced analysis.
- 💼 **Versatility**: Easy to personalize and suitable for other Jupyter Notebook platforms.

---

## 🚀 Quick Start
1. For data from TrackMate and the TrackMate batcher:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab_TrackMate.ipynb)
2. For other software-generated data:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guijacquemet/CellTracksColab/blob/main/Notebook/CellTracksColab.ipynb)

---

## 🖼️ Screenshots
![Figure 1](path_to_figure1_image)
*Caption for Figure 1*

![Figure 2](path_to_figure2_image)
*Caption for Figure 2*

... [Add more figures as needed]

---

## 📊 Test Dataset
Start with our test dataset. Get it [here](https://zenodo.org/record/8413510).

---

## 📁 Data Structure and Requirements
**Note**: **CellTracksColab** does not support track splitting.

Depending on the notebook you use, the data requirement is slightly different. Check our wiki or the notebook for details. In any case, organize your data with our recommended two-tiered folder hierarchy.

- 📁 **Experiments** `[Folder_path]`
  - 🌿 **Condition_1** `[‘condition’ is derived from this folder name]`
    - 🔄 **R1** `[‘repeat’ is derived from this folder name]`
      - 📄 `FOV1.csv`
      - 📄 `FOV2.csv`
    - 🔄 **R2**
      - 📄 `FOV1.csv`
      - 📄 `FOV2.csv`
  - 🌿 **Condition_2**
    - 🔄 **R1**
    - 🔄 **R2**

In this representation, different symbols are used to represent folders and files clearly:

📁 represents the main folder or directory.
🌿 represents the condition folders.
🔄 represents the repeat folders.
📄 represents the individual CSV files.

---

## 📚 [Visit our wiki for more details](wiki_link)

---

## ✍️ Creator
Created by [Guillaume Jacquemet](https://cellmig.org/)

---

## 🤝 Contributing
For contributions, please fill in an [issue](github_issue_link).

---

## 📜 License
**CellTracksColab** is under the MIT License. Read the full text [here](https://opensource.org/licenses/MIT).
