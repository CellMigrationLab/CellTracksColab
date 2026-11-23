# Keep your app up to date

The LabConstrictor template keeps improving. Your repository can receive those improvements, including updates to its GitHub Actions workflows, as a pull request that you review and merge. This is optional and can wait until after your first release.

## How it works

The **Sync with Template Repository** workflow runs every Monday at 09:00 UTC, and you can also start it yourself from the **Actions** tab. When the template has a newer version than your repository, the workflow applies the changes needed to move your repository to that version and opens a pull request titled "Sync template to version" followed by the new version number. You then merge it as described in [Accept a pull request](accept_pull_request.md). When your repository is already on the latest template version, the run ends without opening a pull request.

Template-managed files, such as the copy of the App Centre manifest schema used by the **Check App Centre manifest** workflow, are refreshed by the same pull request. If that check rejects a field the App Centre accepts, your copy of the schema is older than the App Centre: merge the latest sync pull request.

## What you lose without the setup

Opening that pull request needs a token that you create and save in your repository as the secret `LABCONSTRICTOR_SYNC_TOKEN`. Without it, the workflow stops with the error "Synchronization setup required" as soon as a newer template version exists. No update pull request is created, so your repository stays on the template version it has.

## Set it up

Follow [Enable automatic LabConstrictor template updates](template_synchronization.md): create the token, save it as the secret, and renew it when it expires ([Renewing an expired token](template_synchronization.md#renewing-an-expired-token)).

## If it does not work

See [Synchronisation is failing on GitHub Actions](troubleshooting.md#synchronisation-is-failing-on-github-actions), and [Check the status of GitHub workflows](workflow_status.md) for how to read a failed run.

---

<div align="center">

[← Previous](external_code_upload.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](template_synchronization.md)


</div>
