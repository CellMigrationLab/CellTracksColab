# Describe your app: the `labconstrictor-app.yaml` file

`labconstrictor-app.yaml` is a small text file in the root of your repository. It says what your app is, in your own words. You write it once and keep it up to date yourself.

Three things read it:

- **The [App Centre](https://cellmigrationlab.github.io/LabConstrictor-Site/apps/)** builds your app's card and page from it.
- **Your release workflow** reads `platforms` to decide which installers to build.
- **Your installer** passes `hosts` and `host_requirements` to napari, Fiji and QuPath, so they know whether to list your app.

Version numbers and download links are not in the file. They always come from your latest GitHub release.

## Step 1: get the file

The [repository initialiser](initialise_repository.md) asks for a one-sentence summary, the kind of app and the software it also works in. It adds the file to the initialisation pull request. Read it there and edit it if you like.

If you skipped that section, or your repository is older than the file, create `labconstrictor-app.yaml` in the root of your repository and paste the minimal example below. The [Submit page](https://cellmigrationlab.github.io/LabConstrictor-Site/submit/) also has a form that writes the file for you and checks it as you type.

## Step 2: start with the minimal file

These five fields are required. This file is complete and valid:

```yaml
schema: 1
name: My App
summary: Counts nuclei in 2D fluorescence images and saves a table.
kind: analysis
distribution:
  installer: true
```

## Step 3: add what helps people choose your app

A fuller example. Everything after `distribution` is optional.

```yaml
# yaml-language-server: $schema=https://cellmigrationlab.github.io/LabConstrictor-Site/schema/labconstrictor-app.schema.json
schema: 1
name: My App
summary: Counts nuclei in 2D fluorescence images and saves a table.
description: |-
  A few paragraphs of plain text.

  Separate paragraphs with a blank line.
kind: analysis
hosts: [notebook, napari, qupath]
host_requirements:
  napari:
    min_version: 0.1.0-alpha.5
    why: Older plugins do not show the outlines
    required: false
distribution:
  installer: true
  platforms: [windows-x86_64, macos-arm64, linux-x86_64]
  colab:
    - label: Notebook
      url: https://colab.research.google.com/github/you/my-app/blob/main/notebooks/App/App.ipynb
  pip: my-app
status: Available
license: MIT
authors:
  - name: Ada Lovelace
    orcid: 0000-0002-1825-0097
doi: 10.1371/journal.pbio.3002740
inputs:
  - 2D TIFF images, one channel per file
outputs:
  - Label masks and a CSV table
limitations:
  - Not tested on 3D data
quickstart:
  - Open the app and run the Welcome notebook.
categories: [nuclei, counting]
homepage: https://github.com/you/my-app
docs_url: https://github.com/you/my-app#readme
icon: app/logo/logo.png
screenshots:
  - path: docs/screenshot.png
    caption: The result table next to the segmented image
```

The first line lets editors with YAML support (for example VS Code) check the file as you type. Everything is plain text: no HTML and no Markdown.

## The fields

| Field | Required | What it does | Example |
|---|---|---|---|
| `schema` | Yes | The format version. Always `1`. | `1` |
| `name` | Yes | The name people know the app by (2 to 60 characters). | `My App` |
| `summary` | Yes | One sentence for the app card (20 to 200 characters). | `Counts nuclei in 2D images.` |
| `kind` | Yes | What the app is for. One of `analysis`, `segmentation`, `simulation`, `game`, `training`, `utility`. It drives the filter in the App Centre. | `analysis` |
| `distribution` | Yes | How people get the app. At least one of `installer`, `colab`, `pip`. See below. | `installer: true` |
| `hosts` | No | Where the app's tools run: `notebook`, `napari`, `fiji`, `qupath`. See below. | `[notebook, napari]` |
| `host_requirements` | No | The oldest napari, Fiji or QuPath plugin your tools need. See below. | |
| `description` | No | A few paragraphs for the app page (up to 2000 characters). | |
| `status` | No | `Experimental`, `Available` or `Deprecated`. | `Available` |
| `license` | No | The licence, as an SPDX name. | `MIT` |
| `authors` | No | A list of people: `name`, and optionally `orcid`. | `- name: Ada Lovelace` |
| `doi` | No | The DOI of the paper to cite, not of a software archive. The app page shows it as the citation. | `10.1371/journal.pbio.3002740` |
| `inputs`, `outputs`, `limitations` | No | Short lists for the app page: one to six lines of 5 to 200 characters each. | `- 2D TIFF images` |
| `quickstart` | No | Up to six short steps for a first run. | `- Open the app.` |
| `categories` | No | Up to six words a scientist would search for. | `[nuclei, counting]` |
| `homepage`, `docs_url` | No | Web addresses. They must start with `https://`. | |
| `icon` | No | A square PNG in your repository. SVG is not accepted. | `app/logo/logo.png` |
| `screenshots` | No | One to six pictures in your repository (PNG, JPG or GIF), each with a `path` and a `caption`. The caption is also the text for screen readers. | |

The App Centre asks for a few optional fields: `status`, `quickstart`, `screenshots`, a licence, and `inputs` and `outputs` for analysis, segmentation and simulation apps. Without them your app is still listed; its page says "Not stated" or leaves that part out.

### `distribution`: how people get the app

| Key | What it does |
|---|---|
| `installer: true` | Your latest GitHub release carries the installers LabConstrictor builds. |
| `platforms` | The systems you support. Needs `installer: true`. Only these installers are built and offered. |
| `colab` | A list of Google Colab notebooks, each with a `label` and a `url` on `colab.research.google.com`. Only the notebooks you list are shown. |
| `pip` | The name of your package on PyPI. |

`platforms` also decides which installers your releases build. Leave a system out when your app cannot run there. Without `platforms`, every installer is built: `windows-x86_64`, `linux-x86_64`, `macos-arm64` (Apple silicon) and `macos-x86_64` (Intel Macs). `windows-arm64` and `linux-arm64` are accepted, but no installer is built for them.

### `hosts`: where your tools are offered

This field matters only if your app [exposes tools](tools_bridge.md). napari, Fiji and QuPath list your app only if you name them here. Write the hosts you have tested the app in.

- `hosts: [notebook]` means no plugin lists the app. Its tools still work in the notebook and on the command line.
- Without a `hosts` line, every plugin lists the app. Say what you mean instead.
- Names are lower case. The check refuses a misspelt, repeated or capitalised name.

## Plugin versions your tools need (optional)

If your tools rely on a fix or a feature of the napari, Fiji or QuPath plugin that older plugins lack, say so with `host_requirements`. People with an older plugin keep seeing your app, with one sentence saying which plugin to update and where; your app page on the App Centre lists the versions under "Plugin versions".

```yaml
host_requirements:
  qupath:
    min_version: 0.1.0-alpha.7                  # a release tag of the plugin, with or without v
    features: [shapes_owner_v1]                 # optional: a host feature id
    why: Replace keeps your own annotations     # optional: one line shown to the person
  fiji:
    min_version: 0.1.0-alpha.7
```

- Name only `napari`, `fiji` or `qupath`: the notebook ships with your app and needs no plugin.
- Each needs `min_version`, `features`, or both. `required: true` makes an older plugin refuse your tools instead of warning; use it only when they really fail there.
- Write it as a block, as above. The installer passes this file to `labconstrictor-tools register`, which refuses a malformed block (a misspelt host or key, a version like `latest`); `labconstrictor-tools check --module <your>_lc_tools --app-manifest labconstrictor-app.yaml` shows what it read.
- The feature ids and the rule are in [the protocol reference](https://cellmigrationlab.github.io/LabConstrictor-Site/guides/protocol/#host-requirements-which-plugin-an-app-needs).

## The automatic check

The **Check App Centre manifest** workflow runs when you change the file and when you publish a release:

- If the file is missing, it shows a warning and passes.
- If the file is invalid, it fails and says what to fix. This never stops your installers from being built.
- It rejects fields it does not know (this catches typos, so the schema is not loosened); the template keeps its copy of the schema in step with the App Centre, see [Keep your app up to date](keep_app_up_to_date.md).
- It also checks that the icon and screenshots you name exist in the repository.

## Check it yourself

To check the file before you push:

1. **On your computer.** From the root of your repository:

    ```bash
    pip install pyyaml jsonschema
    python .tools/python/check_app_manifest.py
    ```

2. **In your browser.** Paste your values into the form on the [Submit page](https://cellmigrationlab.github.io/LabConstrictor-Site/submit/). It checks them against the same schema.

If your app exposes tools, also run `labconstrictor-tools check --module <your>_lc_tools --app-manifest labconstrictor-app.yaml`. It prints the `hosts` and `host_requirements` it read, which is what the installer will register.

The rules themselves are in the [schema](https://cellmigrationlab.github.io/LabConstrictor-Site/schema/labconstrictor-app.schema.json). Your repository has a copy in `.tools/schema/`, and the check uses that copy.

## Common mistakes

| What the check says, or what you see | Fix |
|---|---|
| A field "is not allowed" or is "unexpected" | The name is misspelt, or it is at the wrong level. For example `platforms` belongs under `distribution`. |
| `summary` is too short or too long | Write one sentence of 20 to 200 characters. |
| `platforms` without `installer` | Add `installer: true`. `platforms` only applies to installers. |
| The icon or a screenshot is "not found" | The path starts at the root of your repository, with no leading `/`. The file must be committed. |
| A Colab link is refused | The address must start with `https://colab.research.google.com/`. |
| `min_version: 0.2` is refused | YAML reads `0.2` as a number. Put it in quotes: `"0.2"`. |
| Your app is missing from napari, Fiji or QuPath | The host is not in `hosts`, or the app was installed before you added it. Publish a release and install it again. |
| The check passes but nothing changes in the App Centre | The App Centre reads the file from your default branch and updates once a day. |
| Text shows `**stars**` or tags | The fields are plain text. Remove Markdown and HTML. |

## Get listed

Having the file does not list your app by itself. To list it, [submit your repository](https://cellmigrationlab.github.io/LabConstrictor-Site/submit/) to the App Centre. You do this once: later releases appear by themselves.

---

<div align="center">

[← Previous](installer_images.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](external_code_upload.md)


</div>
