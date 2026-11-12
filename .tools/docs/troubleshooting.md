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

## Publishing to PyPI fails

The workflow **Publish the Python package to PyPI** (`publish_pypi.yml`) runs at a release only when the repository variable `PYPI_PUBLISH` is `true` or `testpypi` (see [Upload external code](external_code_upload.md#4-publish-the-code-on-pypi-optional)). Its first job checks and builds the package; its second job uploads it. The installers do not depend on it: a failure here leaves the release and its installers as they are.

| What you see | What to do |
|---|---|
| Both jobs are grey (skipped) | The variable is missing or has another value. It must be a **variable** (Settings > Secrets and variables > Actions > **Variables**), not a secret, named `PYPI_PUBLISH`, with the value `true` or `testpypi` exactly. |
| `invalid-publisher: valid token, but no corresponding publisher` | The trusted publisher on pypi.org (or test.pypi.org for `testpypi`) does not match this workflow. Check the five fields: owner, repository, workflow `publish_pypi.yml`, environment `pypi`, and the project name. Then re-run the failed job (Actions > the run > **Re-run failed jobs**). |
| `... isn't allowed to upload to project ...` (403) | The name belongs to another PyPI project that this repository is not a publisher of. If it is yours, add the publisher in that project's **Publishing** page; if not, choose another `name=` in `setup.py` (and `distribution.pip`). |
| `PyPI already has <name> <version>` (or `File already exists`) | That version was uploaded before; a version can be uploaded only once, even after it was deleted. Publish a new release with a higher version. |
| `There is no Python package in src/` | Add your code as a folder with an `__init__.py` in `src/` (see [Upload external code](external_code_upload.md)), or delete the variable `PYPI_PUBLISH`. |
| `... also publishes to PyPI` | Another workflow of the repository uploads to PyPI too. Keep one: delete the variable `PYPI_PUBLISH`, or delete the other workflow. |
| `distribution.pip of labconstrictor-app.yaml is ..., but setup.py names the package ...` | Make the two names the same. |
| `PyPI refuses a package whose dependency is a URL` | `setup.py` lists a dependency as `name @ git+https://...`. Publish that dependency on PyPI and name it with a version, or move the URL to `requirements.txt` (the installer still installs it; the PyPI package leaves it out). |
| `setup.py still has the template placeholder ...` | Replace the placeholder in `setup.py` (the log says which), commit, and publish a new release. |
| The upload job waits | The `pypi` environment has required reviewers (Settings > Environments > pypi): one of them approves the upload in the run. |

After a fix, re-run the failed jobs of the run if nothing in the repository changed (a publisher fixed on pypi.org); otherwise publish a new release. See [the design note of template 0.1.25](https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_25.md).
