# Check the status of GitHub workflows

GitHub Actions runs workflows in your repository: when a notebook or its requirements change (**Updates on notebook change**), when the code in `src/` changes (**Updates on external code change**), when you publish a release (**Build Release Installers**) and every week for template updates (**Sync with Template Repository**, which needs the one-time [`LABCONSTRICTOR_SYNC_TOKEN` setup](template_synchronization.md)).

## 1. Monitor the progress

1. Go to the **Actions** tab of your repository.
2. Find the run that matches your latest change or release.
3. Check its status icon:

![GitHub Actions Status Icons](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/actions_status_icons.png)

## 2. How to debug a failure

1. Click the failed run, then the red step: its log says what went wrong.
   The log usually names the command or the package that failed.
2. If the run has an **Artifacts** section at the bottom of its page, download the log file there (a plain `.log` text file, nothing to unzip). It holds the error, the workflow and the script that failed.
3. You can ask an AI assistant to help you read the log; the file starts with instructions for one. First remove tokens, private paths and anything else you would not share. Check any fix it suggests against the error, then run the workflow again.

For known errors, see [Troubleshooting repository setup](troubleshooting.md).

### Best practice for large updates

The workflows run only on your main branch, so a change is checked once it is merged. For a big update, such as several notebooks or new dependencies, upload one notebook at a time and wait for the green tick before the next: a failure then points to one change.

---

<div align="center">

← Previous &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](troubleshooting.md)


</div>

