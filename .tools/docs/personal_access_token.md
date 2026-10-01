# Generate a GitHub Personal Access Token


> Looking for the token used by the automatic template update workflow? Follow the dedicated [automatic template synchronization guide](template_synchronization.md). It uses a repository-scoped fine-grained token with a different purpose and secret name.

If you want to work with GitHub repositories from your notebook, like cloning, editing, or pushing files, you’ll need a GitHub Personal Access Token (PAT). This token lets you authenticate your API calls and gives you the right permissions. The GIF below shows the steps to create a token.

![Gif of generating a GitHub token](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/GitHub_Access_Token.gif)

To see the full steps, follow the instructions below that match the type of access you want.

## For Read & Write Access

1. Go to https://github.com/ and sign in.  
2. Click your profile icon (top right) ➔ **Settings**  
3. In the left sidebar, select **Developer settings** ➔ **Personal access tokens** ➔ **Tokens (classic)**  
4. On the top right side (at the same level as `Personal access tokens (classic)`), click on  **Generate new token** ➔ **Generate new token (classic)**  
5. Give your token a descriptive **Name** (e.g., `Notebook RW Token`) and set an **Expiration** if desired  
6. Under **Scopes**, check:  
   - **repo**  
     - **contents**  
       > Grants read & write access to repository contents (needed for cloning, editing, and pushing files)
   - *(Optional: To only work with public repositories, select **public_repo** instead of **repo**)*  
7. Click **Generate token** at the bottom  
8. **Copy** your new token right away. You won’t be able to see it again later!  
9. Paste the token into your notebook’s `GitHub-token` field, so API calls authenticate as the chosen owner.  

---

## For Read-Only Access

1. Go to https://github.com/ and sign in.  
2. Click your profile icon (top right) ➔ **Settings**  
3. In the left sidebar, select **Developer settings** ➔ **Personal access tokens** ➔ **Tokens (classic)**  
4. Click **Generate new token (classic)**  
5. For **Resource owner**, select the owner of the repository (your user or organisation)  
6. Give your token a descriptive **Name** (e.g., `Notebook Read-Only Token`) and set an **Expiration** if desired  
7. Under **Scopes**, check:  
   - **repo**  
     - **contents**  
       > Grants **read-only** access to repository contents (needed for cloning or viewing files only)  
   - *(Optional: For public repositories, select **public_repo** instead of **repo**)*  
8. Click **Generate token** at the bottom  
9. **Copy** your new token immediately—you won’t see it again!  
10. Paste the token into your notebook’s `GitHub-token` field for authentication.  


---


<div align="center">

[← Previous](workflow_status.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Next →


</div>
