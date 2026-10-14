# Describe your app for the App Centre

The [LabConstrictor App Centre](https://labconstrictor.cellmig.org/apps/) lists apps built with LabConstrictor, with a page for each one. The App Centre reads a small file from **your** repository, `labconstrictor-app.yaml`, so you describe your app once and keep it up to date yourself. Version numbers and download links always come from your latest GitHub release.

## How you get the file

The [repository initialiser](initialise_repository.md) asks for a one-sentence summary, the kind of app and the software it also works in, and adds the file to the initialisation pull request. Read it in the pull request and edit it if you like. If you skip that section, or your repository was created before it existed, copy the example below to the root of your repository.

## What it contains

```yaml
# yaml-language-server: $schema=https://labconstrictor.cellmig.org/schema/labconstrictor-app.schema.json
schema: 1
name: My App
summary: One sentence that says what it does, for the app card.
description: |-
  A few paragraphs of plain text.

  Separate paragraphs with a blank line.
kind: analysis            # analysis, segmentation, simulation, game, training or utility
hosts: [notebook, napari] # where its tools can also run: notebook, napari, fiji, qupath
status: Available         # optional: Experimental, Available or Deprecated
inputs:                   # optional short lists (1 to 6 items of 5 to 200 characters each)
  - TIFF images, one channel per file
outputs:
  - Label masks and a CSV table
limitations:
  - Not tested on 3D data
distribution:             # at least one way to get the app: installer, colab or pip
  installer: true         # the installers LabConstrictor builds for each release
  platforms: [windows-x86_64, macos-arm64, linux-x86_64]  # optional, needs installer: offer only these installers
  colab:
    - label: Notebook
      url: https://colab.research.google.com/github/you/my-app/blob/main/notebooks/App/App.ipynb
  pip: my-app             # the name on PyPI
icon: app/logo/logo.png   # optional: a square PNG in your repository
```

The first line makes editors with YAML support (for example VS Code) check the file as you type. `inputs`, `outputs` and `limitations` are short summaries for the app page (a list of one to six sentences, not README copy). More optional fields exist (authors, DOI, license, screenshots, a quick start); the full list is in the [schema](https://labconstrictor.cellmig.org/schema/labconstrictor-app.schema.json). Everything is plain text: no HTML or Markdown.

## The automatic check

The **Check App Centre manifest** workflow runs when you change the file and when you publish a release:

- If the file is missing, it shows a warning and passes.
- If the file is invalid, it fails and says what to fix. This never stops your installers from being built.
- It rejects fields it does not know (this catches typos, so the schema is not loosened); the template keeps its copy of the schema in step with the App Centre, see [Keep your app up to date](keep_app_up_to_date.md).
- It also checks that the icon and screenshots you name exist in the repository.

## Get listed

Having the file does not list your app by itself. To list it, [submit your repository](https://labconstrictor.cellmig.org/submit/) to the App Centre.

---

<div align="center">

[← Previous](initialise_repository.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[🏠 Home](README.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[Next →](external_code_upload.md)


</div>
