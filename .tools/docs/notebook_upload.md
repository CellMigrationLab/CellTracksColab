# Add or Update a Notebook

If you want to add or update a notebook in your project repository, just follow these steps to upload it smoothly.

## 1. Prepare your notebook

To ensure your notebook works with the versioning tracking provided by the LabConstrictor application, please follow these three steps to prepare your notebook:

1. First, create a cell at the top of your notebook (see our [JupyterLab guide](https://github.com/CellMigrationLab/docs-hub/blob/main/docs/JupyterLab/JupyterLab_Guide.md) if you need help with that). 
2. Then, copy and paste the code bellow into that cell.
3. Finally, update the `current_version` variable with the version of your notebook (for example, "0.0.2") and the `notebook_name` variable with the name of your notebook file (without the .ipynb extension).

> IMPORTANT NOTE: Each time you change the notebook's logic, update the version number on `current_version` variable so users can see there is a new version.

```
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
            latest_version = config.get(notebook_name, "")

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

Follow the steps in the [Obtain requirements of the notebook](notebook_requirements.md) guide to create the `requirements.yaml` file for your notebook.

## 3. Upload to the repository

1. Open the [LabConstrictor website](https://labconstrictor-form.streamlit.app/). If you see a `Zzzz` message for inactivity, click `Yes, get this app back up!`.
2.  Select **Go to update flow**.
3. Upload your notebook file (`.ipynb`) and the `requirements.yaml` file you created.
4.  Click **Validate submission**.
5. Enter your repository URL and your personal access token.
    * *(Don't have a token? See [How to create a personal access token](../personal_access_token.md))*
6. Click Create pull request.


## 4. Merge and Verify

If you need help merging the pull request, see the [Accept a Pull Request](accept_pull_request.md) guide for instructions. An automatic workflow will check your notebook and its dependencies.
* **Check Status:** See [How to check workflow status](workflow_status.md).
* Failure Protocol: If the workflow fails, for example, because of dependency conflicts, the system will automatically undo the merge. Check the logs, fix the problem in your local notebook or requirements, and upload again.

---

# How to Update an Existing Notebook

Updating a notebook works the same way as adding a new one:

1. Bump the version: In the top cell of your notebook, increase the `current_version` (for example, from '1.0.0' to '1.0.1').
2. Regenerate requirements: Run the bottom cell to make sure `requirements.yaml` matches the new version number.
3. Upload: Use the [LabConstrictor App](https://labconstrictor-form.streamlit.app/) to upload your file.
    * *Ensure the filename matches the existing one in the repository exactly.*

---
    
<div align="center">

[← Previous](notebook_requirements.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](executable_creation.md)


</div>
