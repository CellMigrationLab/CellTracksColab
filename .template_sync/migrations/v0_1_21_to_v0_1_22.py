"""Template 0.1.21 -> 0.1.22: Windows computers with an NVIDIA driver can install TensorFlow apps; the version cell reads the version file.

Design note: https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_22.md

What this changes in an app repository (every step is independent, idempotent and never deletes anything):

1. .tools/python/merge_requirements.py. The GPU requirements file rewrote `tensorflow` to `tensorflow[and-cuda]` for every system. The
   extra pulls nvidia-* wheels that exist only for Linux and carries no platform marker, so the universal lock pinned them for Windows
   too, and `pip install` failed on every Windows computer with an NVIDIA driver (post_install.bat picks the GPU file there). The GPU
   file now has `tensorflow[and-cuda]==X ; sys_platform == "linux"` and `tensorflow==X ; sys_platform != "linux"`. Three known blocks
   are changed (docstring, a new function before build_gpu_lines, the tensorflow branch of build_gpu_lines).
2. .tools/docs/notebook_upload.md. The version cell snippet looked the notebook up at the top level of notebook_latest_versions.yaml,
   which sync_notebook_versions.py writes grouped by folder, so Colab users always read "not listed". The one known line is replaced.
   Notebooks that already contain the old snippet are not edited (see the design note).

Rules this migration keeps: surgical (only known blocks), idempotent (a second run reports no changes), never raises for a customised
file (the sync pull request must still be created), the line endings of each file are kept, nothing is deleted.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


DOCS_URL = "https://github.com/CellMigrationLab/LabConstrictor/blob/main/.tools/docs/migration_0_1_22.md"
TAG = "[migration 0.1.22]"
MERGE = Path(".tools/python/merge_requirements.py")
UPLOAD_DOC = Path(".tools/docs/notebook_upload.md")

# merge_requirements.py: (0.1.21 text, 0.1.22 text), LF.
MERGE_DOC_OLD = (
    "GPU file rewrites `tensorflow` to `tensorflow[and-cuda]` and adds the PyTorch\n"
    "CUDA wheel index when `torch`, `torchvision`, or `torchaudio` are present.\n"
)
MERGE_DOC_NEW = (
    "GPU file rewrites `tensorflow` to `tensorflow[and-cuda]` on Linux only (TensorFlow's CUDA\n"
    "wheels exist only for Linux; other systems keep plain `tensorflow`) and adds the\n"
    "PyTorch CUDA wheel index when `torch`, `torchvision`, or `torchaudio` are present.\n"
)
MERGE_ANCHOR = "def build_gpu_lines(final_lines: list[str]) -> list[str]:\n"
MERGE_FUNCTION = (
    "# TensorFlow's `and-cuda` extra pulls nvidia-* wheels that exist only for Linux (nvidia-nccl-cu12 has never had a Windows or macOS\n"
    '# wheel), and the extra itself carries no platform marker. Without a marker the universal lock of requirements_gpu.txt pins them for\n'
    '# every system, so `pip install` fails on a Windows computer with an NVIDIA driver (the installer picks the GPU file there). TensorFlow\n'
    '# has had no native Windows GPU build since 2.10, so Windows and macOS keep plain `tensorflow` in the GPU file.\n'
    'GPU_TENSORFLOW_PLATFORM = \'sys_platform == "linux"\'\n'
    'NON_GPU_TENSORFLOW_PLATFORM = \'sys_platform != "linux"\'\n'
    '\n'
    '\n'
    'def _with_platform(marker: str, platform: str) -> str:\n'
    '    """Return ` ; <platform>`, combined with the line\'s own marker (`; <marker>`) when it has one."""\n'
    '    existing = marker.split(";", 1)[1].strip() if ";" in marker else ""\n'
    '    return f" ; ({existing}) and {platform}" if existing else f" ; {platform}"\n'
    '\n'
    '\n'
    'def tensorflow_gpu_lines(line: str) -> list[str]:\n'
    '    """Return the GPU-file lines for a tensorflow requirement: `[and-cuda]` on Linux, the plain requirement elsewhere.\n'
    '\n'
    '    A line that is not a tensorflow requirement is returned unchanged (as a one-item list).\n'
    '    """\n'
    '    match = pkg_re.match(line)\n'
    '    if match:\n'
    '        if match.group("name").lower() != "tensorflow":\n'
    '            return [line]\n'
    '        name, extras = match.group("name"), match.group("extras") or ""\n'
    '        spec = f"{match.group(\'op\')}{match.group(\'ver\')}"\n'
    '    else:\n'
    '        match = PLAIN_PKG_RE.match(line)\n'
    '        if not match or match.group("name").lower() != "tensorflow":\n'
    '            return [line]\n'
    '        name, extras, spec = match.group("name"), match.group("extras") or "", ""\n'
    '    marker = match.group("marker") or ""\n'
    '    cuda_extras = merge_requirement_extras(extras, "and-cuda")\n'
    '    return [\n'
    '        f"{name}{cuda_extras}{spec}{_with_platform(marker, GPU_TENSORFLOW_PLATFORM)}",\n'
    '        f"{name}{extras}{spec}{_with_platform(marker, NON_GPU_TENSORFLOW_PLATFORM)}",\n'
    '    ]\n'
    '\n'
    '\n'
)
MERGE_BODY_OLD = (
    '            if package_name == "tensorflow":\n'
    "                line = rewrite_tensorflow_for_gpu(line)\n"
    "            if package_name in TORCH_GPU_PACKAGES:\n"
    "                has_torch_package = True\n"
)
MERGE_BODY_NEW = (
    "            if package_name in TORCH_GPU_PACKAGES:\n"
    "                has_torch_package = True\n"
    '            if package_name == "tensorflow":\n'
    "                gpu_lines.extend(tensorflow_gpu_lines(line))\n"
    "                continue\n"
)
MERGE_DONE = "def tensorflow_gpu_lines("

UPLOAD_OLD = '            latest_version = config.get(notebook_name, "")\n'
UPLOAD_NEW = (
    '            # notebook_latest_versions.yaml groups the notebooks by folder: {"<folder>": {"<notebook>": "<version>"}}\n'
    "            latest_version = next((str(group[notebook_name]) for group in (config or {}).values()\n"
    '                                   if isinstance(group, dict) and notebook_name in group), "")\n'
)
UPLOAD_DONE = "groups the notebooks by folder"


def log(message: str) -> None:
    print(f"{TAG} {message}")


def _load_helpers() -> Any:
    """Reuse the helpers of the 0.1.10 migration (newline-preserving file IO)."""
    path = Path(__file__).resolve().parent / "v0_1_9_to_v0_1_10.py"
    spec = importlib.util.spec_from_file_location("template_sync_helpers_0_1_22", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_merge(text: str, newline: str) -> tuple[str, str]:
    """Return (new text, status) for merge_requirements.py; the code blocks change together or not at all."""
    if MERGE_DONE in text:
        return text, "unchanged: the GPU file already limits tensorflow[and-cuda] to Linux"
    anchor, body_old, body_new = (s.replace("\n", newline) for s in (MERGE_ANCHOR, MERGE_BODY_OLD, MERGE_BODY_NEW))
    if text.count(anchor) != 1 or text.count(body_old) != 1:
        return text, f"SKIPPED: the 0.1.21 build_gpu_lines was not found once (customised file); see {DOCS_URL} for the change to make by hand"
    function = MERGE_FUNCTION.replace("\n", newline)
    text = text.replace(anchor, function + anchor, 1).replace(body_old, body_new, 1)
    doc_old, doc_new = MERGE_DOC_OLD.replace("\n", newline), MERGE_DOC_NEW.replace("\n", newline)
    if text.count(doc_old) == 1:
        text = text.replace(doc_old, doc_new, 1)
    return text, "updated: tensorflow[and-cuda] only on Linux in the GPU file"


def _patch_upload(text: str, newline: str) -> tuple[str, str]:
    if UPLOAD_DONE in text:
        return text, "unchanged: the version cell already reads the grouped version file"
    old = UPLOAD_OLD.replace("\n", newline)
    if text.count(old) != 1:
        return text, f"SKIPPED: the 0.1.21 version lookup line was not found once (customised file). See {DOCS_URL}"
    return text.replace(old, UPLOAD_NEW.replace("\n", newline), 1), "updated: the version cell reads the grouped version file"


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    """Apply both changes (idempotent, surgical, never raises for a customised or missing file)."""
    _ = context
    helpers = _load_helpers()
    changed = 0
    for relative, patch in ((MERGE, _patch_merge), (UPLOAD_DOC, _patch_upload)):
        path = repo_root / relative
        if not path.is_file():
            log(f"{relative}: SKIPPED, file not found in this repository (nothing was created).")
            continue
        try:
            text = helpers.read_text(path)
            new_text, status = patch(text, "\r\n" if "\r\n" in text else "\n")
            if new_text != text:
                helpers.write_text(path, new_text)
                changed += 1
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            status = f"SKIPPED: {exc}; left unchanged. See {DOCS_URL}"
        log(f"{relative}: {status}")
    if changed:
        log(f"done: {changed} file(s) changed. Push a notebook change (or run 'Updates on notebook change') to regenerate the GPU lock, then release a new installer.")
    else:
        log("done: no repository changes were required.")
