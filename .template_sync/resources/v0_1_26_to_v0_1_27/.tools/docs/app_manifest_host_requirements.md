## Plugin versions your tools need (optional)

If your tools rely on a fix or a feature of the napari, Fiji or QuPath plugin that older plugins lack, say so with `host_requirements`. People with an older plugin keep seeing your app, with one sentence saying which plugin to update and where; your app page on the App Centre lists the versions under "Plugin versions".

```yaml
host_requirements:
  qupath:
    min_version: 0.1.0-alpha.6                  # a release tag of the plugin, with or without v
    features: [shapes_owner_v1]                 # optional: a host feature id
    why: Replace keeps your own annotations     # optional: one line shown to the person
  fiji:
    min_version: 0.1.0-alpha.6
```

- Name only `napari`, `fiji` or `qupath`: the notebook ships with your app and needs no plugin.
- Each needs `min_version`, `features`, or both. `required: true` makes an older plugin refuse your tools instead of warning; use it only when they really fail there.
- Write it as a block, as above. The installer passes this file to `labconstrictor-tools register`, which refuses a malformed block (a misspelt host or key, a version like `latest`); `labconstrictor-tools check --module <your>_lc_tools --app-manifest labconstrictor-app.yaml` shows what it read.
- The feature ids and the rule are in the [LabConstrictor-Tools documentation](https://github.com/CellMigrationLab/LabConstrictor-Tools/blob/main/docs/PROTOCOL.md#host-requirements-which-plugin-an-app-needs).
