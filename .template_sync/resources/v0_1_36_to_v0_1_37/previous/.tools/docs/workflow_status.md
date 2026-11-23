# Check the status of GitHub workflows

When you upload a notebook or make a release, GitHub Actions starts a workflow to check your code and build the installers. The template synchronization workflow additionally requires the one-time [`LABCONSTRICTOR_SYNC_TOKEN` setup](template_synchronization.md).

## 1. Monitor the progress

1.  Go to the **Actions** tab in your GitHub repository.
2. You’ll see a list of recent workflows. Find the one that matches your latest commit or release.
3.  Check the status icon next to the workflow name:

![GitHub Actions Status Icons](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/actions_status_icons.png)

## 2. How to debug a failure

If the workflow fails, don’t worry. The system creates an error log that’s ready for you to use.

1.  **Download the Log:**
    * Click on the failed workflow run.
    * Scroll down to the bottom of the page to the **Artifacts** section.
    * Download the log file listed there (it is a plain `.log` text file, nothing to unzip).

2.  **Copy & Paste into AI:**
    * Open the log file in any text editor.
    * **This file is already formatted as a prompt.** It includes all the context and error details you need.
    * Just **copy all the text** and paste it into your preferred AI tool (like ChatGPT, Gemini, or Claude). The AI will explain the error and show you how to fix your notebook or `requirements.yaml`.

---

### Best practice for large updates

If you’re planning a big update, like changing several notebooks or adding complex dependencies, don’t push straight to the main branch.

Instead, **create a new branch**, make your changes there, and check that the workflows pass (✅) before merging into the main branch. This helps keep your live app working.

---

<div align="center">

← Previous &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](troubleshooting.md)


</div>

