# Add or update a notebook

Upload a notebook and its `requirements.yaml` to your repository with the LabConstrictor web form. The form opens a pull request; when you merge it, GitHub checks that the notebook's packages can be installed together.

## What you need

- An initialised repository: see [Initialise your repository](initialise_repository.md).
- Your notebook (`.ipynb`) and its requirements file: see [Generate the requirements of your notebook](notebook_requirements.md).
- A GitHub personal access token (classic, it starts with `ghp_`): see [Generate a GitHub personal access token](personal_access_token.md).

## 1. Prepare your notebook

The first cell of every notebook checks its version, so users see when a newer version exists.

1. Add a code cell at the top of your notebook (see this [JupyterLab guide](https://github.com/CellMigrationLab/docs-hub/blob/main/docs/JupyterLab/JupyterLab_Guide.md) if you need help), and paste the code below into it.
2. Set `current_version` to the version of your notebook (for example `"0.0.2"`) and `notebook_name` to the name of the notebook file without `.ipynb`.

> Each time you change what the notebook does, increase `current_version` so users can see there is a new version.

```python
# @markdown Run this cell to check this notebook's version.

current_version = "0.0.1"
notebook_name = "Notebook_template"  # Replace with the actual notebook name

# First of all check if ipywidgets is installed, if not through an informative error
try:
    import ipywidgets as widgets
except ImportError:
    raise ImportError("ipywidgets is not installed. Please install it using 'pip install ipywidgets' or 'conda install -c conda-forge ipywidgets'.")

from IPython.display import display

try:
    from google.colab import output
    output.enable_custom_widget_manager()
    
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is not installed. Please install it using 'pip install pyyaml' or 'conda install -c conda-forge pyyaml'.")

    import requests
    import base64

    # Create widgets
    github_repo = widgets.Text(
        value="",
        description="*GitHub Repository URL:",
        placeholder="e.g https://github.com/username/repository",
        layout=widgets.Layout(width="70%"),
        style={'description_width': '150px'},
    )

    github_token = widgets.Password(
        value="",
        description="*Personal Access Token:",
        placeholder="[Disabled] e.g. ghp_XXXXXXXXXXXXXXXXXXXX",
        disabled=True,
        layout=widgets.Layout(width="70%"),
        style={'description_width': '150px'},
    )

    test_button = widgets.Button(
        description="Test Connection",
        button_style='success',
    )

    output_area = widgets.HTML()

    # By default the repository is considered public
    if "repository_is_private" not in globals():
        repository_is_private = False
    elif repository_is_private:
        github_token.disabled = False
        github_token.placeholder = "e.g. ghp_XXXXXXXXXXXXXXXXXXXX"

    def on_test_button_clicked(b):
        global repository_is_private
        
        # Check that the GitHub repository URL is provided
        if not github_repo.value:
            output_area.value = "⚠️ Please provide a GitHub repository URL."
            return
        
        # Get the owner and repo name from the URL
        repo_url = github_repo.value.rstrip('/')
        github_owner, github_repo_name = repo_url.split('/')[-2:]
        
        # Online version checking file path
        version_file_path = "notebooks/notebook_latest_versions.yaml"
        version_url = f"https://api.github.com/repos/{github_owner}/{github_repo_name}/contents/{version_file_path}"

        # Do an initial request to check if the repository is public
        if not repository_is_private:
            version_response = requests.get(version_url)
            if version_response.status_code == 404:
                repository_is_private = True
                github_token.disabled = False
                github_token.placeholder = "e.g. ghp_XXXXXXXXXXXXXXXXXXXX"
                output_area.value = f"⚠️ We have detected that the repository might be private. Please provide a Personal Access Token and click 'Test Connection' again."
                return
        else:
            headers = {"Accept": "application/vnd.github.v3+json"}
            if not github_token.value:
                output_area.value = "⚠️ Personal Access Token is required for private repositories."
                return
            headers["Authorization"] = f"token {github_token.value}"

            version_response = requests.get(version_url, headers=headers)

        # Check the response status
        if version_response.status_code == 200:
            content = version_response.json()['content']
            decoded_content = base64.b64decode(content).decode('utf-8')
            config = yaml.safe_load(decoded_content)
            # notebook_latest_versions.yaml groups the notebooks by folder: {"<folder>": {"<notebook>": "<version>"}}
            latest_version = next((str(group[notebook_name]) for group in (config or {}).values()
                                   if isinstance(group, dict) and notebook_name in group), "")

            output_area.value = (f"<b>Notebook version:</b> `{current_version}`<br>"
                                f"<b>Latest version available:</b> `{latest_version}`<br>")

            if latest_version == "":
                output_area.value += "⚠️ This notebook is not listed in the version file.<br>"
            elif current_version == latest_version:
                output_area.value += "✅ This notebook is up-to-date.<br>"
            else:
                output_area.value += f"⚠️ A new version of this notebook is available."
        else:
            output_area.value += "⚠️ Could not retrieve the version file.<br>"

    test_button.on_click(on_test_button_clicked)
    # Widget layout
    widget_box = widgets.VBox([github_repo, github_token, test_button, output_area])
    display(widget_box)
except ImportError:
    display(widgets.HTML('To check the notebook version, please go to the <a href="../Welcome.ipynb" target="_blank" style="color: #0066cc; text-decoration: underline;">Welcome notebook</a>.'))
```


## 2. Generate the requirements file

Create the `requirements.yaml` file of your notebook: see [Generate the requirements of your notebook](notebook_requirements.md).

## 3. Upload to the repository

1. Open the [LabConstrictor form](https://labconstrictor-form.streamlit.app/). If it shows a `Zzzz` message, click `Yes, get this app back up!` and wait.
2. Click **Upload/Update notebook**.
3. Upload your notebook file (`.ipynb`) and its requirements file.
4. Click **➕ Validate & Add Submission**. If the notebook and its requirements pass the checks, the notebook appears in the **📃 Submission list**. Repeat steps 3 and 4 to add more notebooks to the same pull request.
5. Under **⬆️ Upload it to GitHub**, enter your repository's address and your personal access token.
6. Click **🚀 Create pull request**.

## 4. Merge the pull request

The form opens a pull request titled "Upload/Update notebook" followed by the notebook's name. Merge it: see [Accept a pull request](accept_pull_request.md). The merge starts the workflow **Updates on notebook change**, which merges the requirements of all notebooks, hides the code cells (unless you disabled it) and checks that the packages can be installed.

## Check that it worked

- In the **Actions** tab, the run of **Updates on notebook change** has a green tick ([how to read a run](workflow_status.md)).
- Your notebook is in `notebooks/<notebook name>/`, next to its `requirements.yaml`.
- `notebooks/notebook_latest_versions.yaml` lists the notebook with its `current_version`.

You can now [create the installers](executable_creation.md).

## If something does not work

- **The form refuses the requirements file.** The message names the line to fix. Each dependency must be `package==version`: see [Generate the requirements of your notebook](notebook_requirements.md).
- **The workflow failed and your merge was undone.** When the merged requirements cannot be installed, the workflow reverts your merge automatically. Open the failed run, find the package that failed in the log, fix the notebook's requirements and upload again.
- **The workflow stops with conflicting versions.** Notebooks ask for different versions of the same package; nothing is installed and the requirements files are not changed. See [When notebooks ask for different versions of a package](notebook_requirements.md#when-notebooks-ask-for-different-versions-of-a-package).
- **The version check says "This notebook is not listed in the version file".** `notebook_name` in the first cell does not match the notebook's file name.

---

## How to update an existing notebook

Updating a notebook works the same way as adding one:

1. In the first cell of your notebook, increase `current_version` (for example from `1.0.0` to `1.0.1`).
2. If the notebook now uses other packages or versions, generate its requirements file again.
3. Upload the notebook and its requirements file with the [LabConstrictor form](https://labconstrictor-form.streamlit.app/), as above. Keep the notebook's file name exactly the same: a notebook with the same name replaces the old one.

---
    
<div align="center">

[← Previous](notebook_requirements.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](executable_creation.md)


</div>
