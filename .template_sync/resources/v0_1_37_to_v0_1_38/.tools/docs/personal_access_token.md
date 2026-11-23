# Generate a GitHub personal access token

> Looking for the token used by the automatic template update workflow? Follow the dedicated [automatic template synchronization guide](template_synchronization.md). It uses a repository-scoped fine-grained token with a different purpose and secret name.

A GitHub personal access token lets LabConstrictor act on your repositories as you. You need one for:

- the [repository initialiser](initialise_repository.md) and the [notebook upload form](notebook_upload.md): they open a pull request in your repository;
- the Welcome notebook, and the version check in the first cell of each notebook, of an installed app whose repository is private: they read the newest notebook versions from the repository (in the Welcome notebook, enter the token and click **Load Private GitHub Repo**).

The forms accept only a classic token, which starts with `ghp_`, and reject a fine-grained token (it starts with `github_pat_`). A classic token needs the **repo** scope to write to a repository and to read a private one.

![Gif of generating a GitHub token](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/GitHub_Access_Token.gif)

## Create the token

1. Go to https://github.com/ and sign in.
2. Click your profile icon (top right) ➔ **Settings**.
3. In the left sidebar, select **Developer settings** ➔ **Personal access tokens** ➔ **Tokens (classic)**.
4. Click **Generate new token** ➔ **Generate new token (classic)**.
5. Give your token a descriptive **Note** (for example `LabConstrictor forms`) and set an **Expiration**. We recommend one: an expired token stops working by itself if you forget it.
6. Under **Scopes**, tick **repo**. If all your repositories are public, **public_repo** is enough.
7. Click **Generate token** at the bottom.
8. **Copy** your new token right away. GitHub shows it only once.
9. Paste it where it is asked for: the **Personal Access Token** field of the repository initialiser or of the notebook upload form, or the **Personal Access Token** field of the Welcome notebook or of a notebook's first cell.

A classic token with the **repo** scope can read and write all your repositories: keep it private, and do not save it in a notebook or a file in your repository.

## Renew or revoke the token

Open **Settings** ➔ **Developer settings** ➔ **Personal access tokens** ➔ **Tokens (classic)**:

- **Renew:** click the token's name, then **Regenerate token**, choose a new expiration and copy the new token. The old one stops working.
- **Revoke:** click **Delete** next to the token. Do this as soon as you no longer need it, or if it may have been seen by someone else.


---


<div align="center">

[← Previous](troubleshooting.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Next →


</div>
