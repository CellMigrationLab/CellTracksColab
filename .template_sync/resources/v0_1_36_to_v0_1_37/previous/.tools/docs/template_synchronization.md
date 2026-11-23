# Enable automatic LabConstrictor template updates

LabConstrictor periodically checks the main template for improvements. When an update is available, the **Sync with Template Repository** workflow prepares the changes in a pull request so you can review and merge them safely.

You will need to do a one-time GitHub setup to create a token that allows LabConstrictor to update your repository. This token described below is limited to the repository you select. LabConstrictor does not need access to your other repositories.

## What do I need to do?

Follow the same instructions that are given in the following video and instructions below.

## Step 1: Create the synchronization token

![GIF showing the token creation](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Synchronisation_Token_Creation.gif)

Follow steps on the GIF or click on [this link](https://github.com/settings/personal-access-tokens/new?name=LabConstrictor+Sync&description=Allows+LabConstrictor+to+prepare+template+update+pull+requests.&expires_in=365&contents=write&pull_requests=write&workflows=write) and follow the steps below:

1. Under **Resource owner**, select the owner of your LabConstrictor repository.
2. Under **Repository access**, select **Only select repositories**.
3. Select your LabConstrictor repository.
4. Choose an expiration period that suits you. The prefilled suggestion is 365 days.
5. Confirm these repository permissions:
   - **Contents:** Read and write
   - **Pull requests:** Read and write
   - **Workflows:** Read and write
6. Select **Generate token**.
7. Copy the generated token.

> GitHub shows the complete token only once. Keep the page open until you have saved it in Step 2.

## Step 2: Save the token as a repository secret

![GIF showing the secret creation](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/Synchronisation_Secret_Creation.gif)

1. Open your LabConstrictor repository.
2. Select **Settings**.
3. In the left sidebar, select **Secrets and variables**, then **Actions**.
4. Select **New repository secret**.
5. Enter this exact name:

   ```text
   LABCONSTRICTOR_SYNC_TOKEN
   ```

6. Paste the token into the **Secret** field.
7. Select **Add secret**.

> GitHub encrypts repository secrets. The token will not be displayed in the repository or in normal workflow logs.

## Step 3 (situational): One-time step for repositories created before template version 0.1.12

An older synchronization workflow does not yet reference `LABCONSTRICTOR_SYNC_TOKEN`. If you already saw an error saying that a GitHub App cannot create or update a workflow without `workflows` permission, make this small edit once:

1. Open [`.github/workflows/sync_template.yml`](../../.github/workflows/sync_template.yml) in your repository.
2. Select the pencil icon to edit the file.
3. Copy and paste all the lines from the [LabConstrictor original workflow](https://github.com/CellMigrationLab/LabConstrictor/blob/main/.github/workflows/sync_template.yml).

4. Select **Commit changes** and commit directly to the default branch.
5. Run **Sync with Template Repository** again.

The migration will then install the complete updated workflow automatically. This manual edit is only needed once for repositories whose existing workflow predates token integration.

## Renewing an expired token

When the token expires:

1. Create a replacement token using Step 1.
2. Open **Settings → Secrets and variables → Actions**.
3. Select `LABCONSTRICTOR_SYNC_TOKEN`.
4. Select **Update secret** and paste the replacement token.
5. Run synchronization again.

You do not need to edit the workflow again.

---

<div align="center">

[← Create repository](create_repository.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[🏠 Documentation home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[Initialise repository →](initialise_repository.md)

</div>
