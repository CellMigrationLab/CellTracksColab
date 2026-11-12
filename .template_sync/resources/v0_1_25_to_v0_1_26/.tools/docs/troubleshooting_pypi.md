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
