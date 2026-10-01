# Hiding Code Cells in Jupyter Notebooks

LabConstrictor gives the option to hide code cells in Jupyter notebooks, making them cleaner and avoiding overwhelming users with too much code. 

## How is this for?

This feature is only useful if your notebooks don't require any code modifications by the user. For example, if your notebook is purely for data visualisation and analysis, or if interaction with the user is done through widgets, hiding code cells can enhance the user experience. If your notebook requires users to modify code cells, it's best to keep them visible.

## How is this different from collapsing cells?

JupyterLab already provides a way of collapsing the cell by clicking the blue bar on the left side of the cell. However, this method uncollapses the cell when the user clicks anywhere inside the cell, which can be inconvenient for users to chose the cell they want to run.

This feature avoids that problem by providing a dedicated button to show or hide all code cells in the notebook and allowing the user to freely click the code cell. Additionally, when hiding code cells using this feature, a play button appears next to each code cell, allowing users to run code cells more intuitively.

## How does the play button work?

Depending on the versions of JupyterLab and ipywidgets you use, when code cells are hidden, a play button might show up next to each code cell:

![Play Button vs Missing](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Notebook_Cell_Button.png)

Clicking this button runs the corresponding code cell without revealing the code. Whether or not the play button appears, you can always run a code cell by selecting it and clicking the play button in the top toolbar:

![Toolbar Play Button](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Notebook_Play_Button.png)

## How to hide or show code cells?

Depending on your intialisation your code cells will be hidden by default or shown by default. In either case, the visibility of code cells can be toggled using the `Show/Hide Code` button in the top toolbar:

![Show Hide Code Button](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Notebook_ShowHide_Button.png)
