# Troubleshooting repository setup

Problems with your repository on GitHub. For problems with an installed app on your computer, see [Troubleshooting an installed app](troubleshooting_installed_app.md).

## Synchronisation is failing on GitHub Actions

The **Sync with Template Repository** workflow needs the repository secret `LABCONSTRICTOR_SYNC_TOKEN`, because template updates can change files in `.github/workflows/`. The setup is in [Keep your app up to date](template_synchronization.md).

### "Synchronization setup required": the token is missing

Create the token and save it under **Settings → Secrets and variables → Actions** with this exact name:

```text
LABCONSTRICTOR_SYNC_TOKEN
```

### GitHub refuses to update a workflow file

If the error says GitHub refuses to create or update a workflow without `workflows` permission, check that the token has all of these repository permissions:

- **Contents:** Read and write
- **Pull requests:** Read and write
- **Workflows:** Read and write

Repositories created before template version `0.1.12` also need the one-time workflow edit described in the synchronization guide, so that the workflow uses the secret.

### Bad credentials or authentication failed

The token has expired or was revoked. Create a new token and update the `LABCONSTRICTOR_SYNC_TOKEN` secret: see [Renewing an expired token](template_synchronization.md#renewing-an-expired-token).

## The release stops at "check_placeholders.py"

Before it builds the installers, the release workflow checks that no template placeholder (`PROJECT_NAME`, `PYTHON_PROJ_NAME`, `UNDERSCORED_PROJECT_NAME`, ...) is left in a file the installer ships. An installer with one would fail on your users' computers (for example `No module named 'PYTHON_PROJ_NAME'` at the end of the installation).

The log of the failed step lists each file and line. Replace the placeholder there with your app's value: `PYTHON_PROJ_NAME` is the Python package of the app (usually its folder in `src/`), `PROJECT_NAME` the app name. For `app/menuinst/Welcome.ipynb`, edit `.tools/templates/Welcome_template.ipynb` instead: the Welcome notebook is made from it at every release. Commit, then publish the release again.

## The release stops at "Installers to build"

The release reads `distribution.platforms` in `labconstrictor-app.yaml` to decide which installers to build. It stops when the file is not valid YAML, when `platforms` is not a list, or when it lists none of the systems the release can build: `windows-x86_64`, `linux-x86_64`, `macos-arm64` and `macos-x86_64`. The log of the step says which. Fix `labconstrictor-app.yaml` (see [Describe your app for the App Centre](app_manifest.md)), commit, then publish the release again.

## "The notebooks ask for incompatible package versions"

The **Run merge script** step of **Updates on notebook change**, or the `merge_requirements.py --check` step of a release, stops when notebooks pin a package to different versions, or ask for versions that no single version meets. Its log lists the package, each notebook and its requirement. Make the notebooks agree, or decide in `requirements_overrides.yaml` which requirement every notebook gets and why, then push it: the workflow runs again. See [When notebooks ask for different versions of a package](notebook_requirements.md#when-notebooks-ask-for-different-versions-of-a-package).

If a template update created `requirements_overrides.yaml` for you, its entries keep the versions your app installed until then (the newest of the pins). Check each one with every notebook that uses the package, and replace its reason with yours.

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

After a fix, re-run the failed jobs of the run if nothing in the repository changed (a publisher fixed on pypi.org); otherwise publish a new release.

---

<div align="center">

[← Previous](workflow_status.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](personal_access_token.md)


</div>
