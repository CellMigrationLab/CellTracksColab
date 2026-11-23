# Create installer files

After uploading your notebooks to the repository, follow these steps to create executable files for sharing.

## Please check before proceeding

Open the Actions tab in your repository and check that no workflows are running. If one is in progress, wait until it finishes. Starting a release while other jobs are running may cause build conflicts.

## 1. Draft a new release

1. Go to your repository's main page.
2. Click on Releases (usually on the right sidebar).
3. Click `Create a new release` and provide the following information:
   - **Tag**: Click `Create a new tag` and enter a version number, for example, `0.0.1`.
      > **IMPORTANT**: Make sure that it follows a semantic versioning format (MAJOR.MINOR.PATCH) e.g. `1.0.0`, `1.2.3`, `2.1.0`. A `v` in front (`v1.0.0`) also works: the installers are named with the version without it.
   - **Release title**: Enter a title for your release, such as `Initial release`.
   - **Description** (optional): Enter a description for your release, for example, `This is the initial release of my project.`

## 2. Publish the release

Once you have entered all the required information, click `Publish release`. This action will start the automated workflow to create installer executable files for your notebooks. To monitor the process, see [How to check the automatic workflow status](workflow_status.md).

When the automated workflow completes, open your repository's **Releases** page, open the release you published and download the installers from its **Assets** list.

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
