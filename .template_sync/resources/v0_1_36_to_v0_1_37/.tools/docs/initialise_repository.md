# Initialise your repository

The repository initialiser, a web form, fills your repository with your app's name, version, description and images. It opens a pull request in your repository; you merge it to finish.

## What you need

- A repository created from the template: see [Create your repository](create_repository.md).
- A name for your app (for example `CellAnalyzer`). We recommend the name of your GitHub repository. It may contain letters, numbers, spaces, `_`, `-` and `.`, from 3 to 50 characters.
- A first version number in the form `MAJOR.MINOR.PATCH` (for example `0.0.1`).
- A GitHub personal access token (classic, it starts with `ghp_`) that can write to the repository: see [Generate a GitHub personal access token](personal_access_token.md). The form refuses a fine-grained token (`github_pat_...`).
- Recommended: a one-sentence summary of what your app does, its kind, and the software it also works in. They go into the description file for the App Centre ([Describe your app for the App Centre](app_manifest.md)).
- Optional: PNG images for the installer. See [Customise the installer images](installer_images.md).

## Fill in the form

1. Open the [repository initialiser](https://labconstrictor-form.streamlit.app/). If it shows a `Zzzz` message, click `Yes, get this app back up!` and wait.
2. In the **Initialise GitHub repository** section, click **Start initialisation**.
3. Fill in **Project name** and **Initial project version**. Tick **Disable automatic code hiding in notebooks** only if users must edit your notebooks' code (see [Hide code cells in notebooks](code_hiding.md)).
4. Under **Describe your app (recommended)**, write the summary, choose the kind of app and the software it also works in.
5. Optionally upload the icon, welcome and header images.
6. Click **Validate submission**. If something is missing, the form lists what to fix.
7. Under **⬆️ Upload it to GitHub**, enter your repository's address (for example `https://github.com/your-lab/CellAnalyzer`) and your token, then click **🚀 Create pull request**.

## Accept the pull request

The form opens a pull request titled "Pull request for <your app> initialisation". Merge it: see [Accept a pull request](accept_pull_request.md).

## Check that it worked

After you merge the pull request:

- Your repository's main page shows a new README titled with your app's name.
- These guides are listed in `.tools/docs/README.md`: the main README of the template moved there, and the new README links to it.
- `construct.yaml` shows your app's name and version instead of `PROJECT_NAME` and `VERSION_NUMBER`.

## If something does not work

- **The form says the token is fine-grained or not valid.** Create a classic token that starts with `ghp_` ([how](personal_access_token.md)).
- **The form says the repository address is not valid.** Use the form `https://github.com/<owner>/<repository>`.
- **"Failed to push changes" or "Failed to create pull request".** Check that the token has the **repo** scope and belongs to an account that can write to the repository, and that the repository's default branch is `main`.
- **No pull request appears.** Look at the messages at the bottom of the form: they say which step failed. You can submit the form again.

For other problems with your repository, see [Troubleshooting repository setup](troubleshooting.md).


---

<div align="center">

[← Previous](create_repository.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](accept_pull_request.md)


</div>
