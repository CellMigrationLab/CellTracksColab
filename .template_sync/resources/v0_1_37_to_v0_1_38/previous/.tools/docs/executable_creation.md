# Create installer files

Publishing a GitHub release builds the installers of your app for Windows (`.exe`), macOS (`.pkg`, Apple Silicon and Intel) and Linux (`.sh`) and attaches them to the release.

## What you need

- At least one notebook uploaded and merged: see [Add or update a notebook](notebook_upload.md).
- A version number for the release, in the form `MAJOR.MINOR.PATCH` (for example `0.0.1`).

## Please check before proceeding

Open the **Actions** tab of your repository and check that no workflow is running. If one is in progress, wait until it finishes: a release started while another workflow runs may build from files that are about to change.

## 1. Draft a new release

1. Go to your repository's main page.
2. Click **Releases** (in the right sidebar).
3. Click **Create a new release** and fill in:
   - **Tag**: click **Create a new tag** and enter the version number, for example `0.0.1`. It must have the form `MAJOR.MINOR.PATCH` (`1.0.0`, `1.2.3`). A `v` in front (`v1.0.0`) also works: the installers are named with the version without it.
   - **Release title**: for example `Initial release`.
   - **Description** (optional): what is new in this release.

## 2. Publish the release

Click **Publish release**. This starts the workflow **Build Release Installers**. It sets the version in your repository's files, then builds one installer per system.

## Check that it worked

- In the **Actions** tab, the run of **Build Release Installers** has a green tick ([how to read a run](workflow_status.md)).
- Open your repository's **Releases** page and the release you published: its **Assets** list holds the installers, named `<App>-<version>-<System>` (for example `MyApp-0.0.1-Windows-x86_64.exe`).
- `.tools/docs/download_executable.md` in your repository now has download buttons for this release.

## If something does not work

- **The run stops at "check_placeholders.py", "Installers to build" or with conflicting package versions.** See [Troubleshooting repository setup](troubleshooting.md): each of these has its own section.
- **Another step failed.** Open the failed run and read the log of the red step ([how](workflow_status.md)). After you fix the problem, delete the release and its tag, then publish the release again.
- **An installer is missing for one system.** The release builds only the systems listed under `distribution.platforms` in `labconstrictor-app.yaml` (all of them when it is not set): see [Describe your app for the App Centre](app_manifest.md).

## 3. After the release

- Try your installer yourself: [Install an app](install.md), then [Use notebooks after installation](notebook_usage.md).
- List your app in the App Centre: [Describe your app for the App Centre](app_manifest.md).
- Receive future template improvements as pull requests: [Keep your app up to date](keep_app_up_to_date.md).


---

<div align="center">

[← Previous](notebook_upload.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](installer_images.md)


</div>
