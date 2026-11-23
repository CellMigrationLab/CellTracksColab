# Enable automatic LabConstrictor template updates

The **Sync with Template Repository** workflow checks the LabConstrictor template every week. When a newer template version exists, it opens a pull request in your repository with the changes, for you to review and merge.

To open that pull request, the workflow needs a token that you create once and save in your repository. The token gives access only to the repository you select, not to your other repositories.

## What you need

- Admin access to your app's repository on GitHub (to add a secret).

## Step 1: Create the synchronization token

![GIF showing the token creation](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Synchronisation_Token_Creation.gif)

Open [this link](https://github.com/settings/personal-access-tokens/new?name=LabConstrictor+Sync&description=Allows+LabConstrictor+to+prepare+template+update+pull+requests.&expires_in=365&contents=write&pull_requests=write&workflows=write), which fills in most of the form, then:

1. Under **Resource owner**, select the owner of your repository.
2. Under **Repository access**, select **Only select repositories**, then select your repository.
3. Choose an expiration period. The link suggests 365 days.
4. Check these repository permissions:
   - **Contents:** Read and write
   - **Pull requests:** Read and write
   - **Workflows:** Read and write
5. Select **Generate token**.
6. Copy the token.

> GitHub shows the token only once. Keep the page open until you have saved it in Step 2.

## Step 2: Save the token as a repository secret

![GIF showing the secret creation](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Synchronisation_Secret_Creation.gif)

1. Open your repository.
2. Select **Settings**.
3. In the left sidebar, select **Secrets and variables**, then **Actions**.
4. Select **New repository secret**.
5. Enter this exact name:

   ```text
   LABCONSTRICTOR_SYNC_TOKEN
   ```

6. Paste the token into the **Secret** field.
7. Select **Add secret**.

GitHub encrypts repository secrets: the token is not shown in the repository or in the workflow logs.

## Check that it worked

Open the **Actions** tab, select **Sync with Template Repository** and click **Run workflow**. The run ends with a green tick. If a newer template version exists, a pull request titled "Sync template to version ..." appears in the **Pull requests** tab: merge it as described in [Accept a pull request](accept_pull_request.md).

## Repositories created before template version 0.1.12

The older synchronization workflow does not use `LABCONSTRICTOR_SYNC_TOKEN`. If a run failed with an error saying that a GitHub App cannot create or update a workflow without `workflows` permission, edit the workflow once:

1. Open [`.github/workflows/sync_template.yml`](../../.github/workflows/sync_template.yml) in your repository.
2. Select the pencil icon to edit the file.
3. Replace all its lines with those of the [template's workflow](https://github.com/CellMigrationLab/LabConstrictor/blob/main/.github/workflows/sync_template.yml).
4. Select **Commit changes** and commit directly to the default branch.
5. Run **Sync with Template Repository** again.

The next template update then keeps the workflow up to date by itself.

## Renewing an expired token

When the token expires, runs fail with an authentication error such as "Bad credentials". Then:

1. Create a new token as in Step 1.
2. Open **Settings → Secrets and variables → Actions**.
3. Select `LABCONSTRICTOR_SYNC_TOKEN`.
4. Select **Update secret** and paste the new token.
5. Run **Sync with Template Repository** again.

## If something does not work

See [Synchronisation is failing on GitHub Actions](troubleshooting.md#synchronisation-is-failing-on-github-actions).

---

<div align="center">

[← Previous](keep_app_up_to_date.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Next →](notebook_portability.md)

</div>
