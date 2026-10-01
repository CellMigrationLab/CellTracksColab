# Troubleshooting

## App still appears in the Windows Installed apps list after uninstalling

On Windows, it may happen that, after uninstalling the app from `Settings > Apps > Installed apps` as explained in the [uninstalling guide](./download_executable.md), the app still appears in the list and does not allow you to uninstall it again, showing a message like:

![Uninstall error message](https://github.com/CellMigrationLab/LabConstrictor/blob/doc_source/troubleshooting/Not_Uninstall.png)

The solution is to go to the Control Panel and uninstall it from there. To do this:

1. Open the Control Panel. You can search for it in the Windows search bar.
2. Go to `Programs > Programs and Features`.
3. Find the LabConstrictor-based app in the list, right-click on it, and select `Uninstall`.
4. A message may appear saying something like: "An error occurred while trying to uninstall xxx. It might have already been uninstalled." Click `Yes` to confirm.

This should remove the app even if the previous error message appears.

## Synchronisation is failing on GitHub Actions

The **Sync with Template Repository** workflow needs the encrypted repository secret `LABCONSTRICTOR_SYNC_TOKEN` because template migrations can update files in `.github/workflows/`. Follow the [automatic template synchronization guide](template_synchronization.md) for the complete beginner-friendly setup.

### The synchronization token is missing

Create the token and save it under **Settings → Secrets and variables → Actions** with this exact name:

```text
LABCONSTRICTOR_SYNC_TOKEN
```

### GitHub refuses to update a workflow file

If the error says GitHub is refusing to create or update a workflow without `workflows` permission, ensure the token has all of these repository permissions:

- **Contents:** Read and write
- **Pull requests:** Read and write
- **Workflows:** Read and write

Repositories created before template version `0.1.12` also need the one-time workflow edit described in the synchronization guide so the existing workflow starts using the secret.

### Bad credentials or authentication failed

The token may have expired or been revoked. Create a replacement token and update the existing `LABCONSTRICTOR_SYNC_TOKEN` secret.

## JupyterLab does not start because Windows reports an ASN.1 certificate error

LabConstrictor-based installers include a TLS-resilient launcher. It attempts
native operating-system certificate verification before JupyterLab imports
Tornado and provides a verified CA-bundle fallback when a malformed Windows
certificate-store entry cannot be parsed. SSL and hostname verification are
never disabled.

The constructor environment intentionally does not include
`pip-system-certs`. Its automatic Python startup hook can fail before the
application launcher has an opportunity to select the verified fallback.
Post-install pip commands instead use an explicit CA bundle while preserving
certificate verification.

Diagnostic information is written to `launcher_debug.log` and
`menuinst_debug.log` in the installation directory. Maintainers can run the
same preflight used by the installer with:

```text
<installation-prefix>\python.exe <installation-prefix>\CellTracksColab\launch_jupyter.py --self-test
```

The self-test reports the project-specific CA override variable. It is derived
from the application name by replacing punctuation and spaces with underscores
and converting it to uppercase, for example `MY_PROJECT_CA_BUNDLE`. All
LabConstrictor-based applications also accept the shared
`LABCONSTRICTOR_CA_BUNDLE` variable. Set either variable to an organization PEM
bundle before installation or launch when a private certificate authority is
required.
