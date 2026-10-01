from __future__ import annotations

from pathlib import Path
from typing import Any


POST_INSTALL_BAT_PATH = Path("app/bash_bat_scripts/post_install.bat")


def detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def update_post_install_bat(repo_root: Path) -> bool:
    path = repo_root / POST_INSTALL_BAT_PATH
    if not path.exists():
        print(f"Skipping missing file: {POST_INSTALL_BAT_PATH}")
        return False

    original_text = read_text(path)
    if "CALL :detect_nvidia_smi" in original_text and ":detect_nvidia_smi" in original_text:
        return False

    newline = detect_newline(original_text)
    lines = original_text.splitlines(keepends=True)
    changed = False

    selected_index = None
    install_index = None
    endlocal_index = None

    for index, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if stripped == 'SET "SELECTED_REQUIREMENTS=%BASE_REQUIREMENTS%"':
            selected_index = index
        elif stripped == '"%PREFIX%\\python.exe" -m pip install -r "%SELECTED_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"':
            install_index = index
        elif stripped == "ENDLOCAL":
            endlocal_index = index

    if selected_index is None or install_index is None or install_index <= selected_index:
        raise ValueError(f"Unable to find GPU requirements selection block in {POST_INSTALL_BAT_PATH}")

    replacement_block = [
        'SET "SELECTED_REQUIREMENTS=%BASE_REQUIREMENTS%"',
        'SET "NVIDIA_SMI="',
        "",
        'IF EXIST "%GPU_REQUIREMENTS%" (',
        "    CALL :detect_nvidia_smi",
        "    IF DEFINED NVIDIA_SMI (",
        '        echo NVIDIA GPU utility detected at "%NVIDIA_SMI%", installing GPU requirements from "%GPU_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
        '        SET "SELECTED_REQUIREMENTS=%GPU_REQUIREMENTS%"',
        "    ) ELSE (",
        '        echo NVIDIA GPU not detected, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
        "    )",
        ") ELSE (",
        '    echo GPU requirements file not found, installing CPU requirements from "%BASE_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
        ")",
        "",
        'echo Installing requirements from "%SELECTED_REQUIREMENTS%" >> "%PREFIX%\\menuinst_debug.log"',
    ]

    replacement_lines = [line + newline for line in replacement_block]
    lines[selected_index:install_index] = replacement_lines
    changed = True

    if endlocal_index is None:
        raise ValueError(f"Unable to find ENDLOCAL in {POST_INSTALL_BAT_PATH}")

    helper_lines = [
        "ENDLOCAL",
        "GOTO :EOF",
        "",
        ":detect_nvidia_smi",
        "FOR %%P IN (",
        '    "%ProgramFiles%\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"',
        '    "%ProgramW6432%\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"',
        '    "%SystemRoot%\\System32\\nvidia-smi.exe"',
        '    "%SystemRoot%\\Sysnative\\nvidia-smi.exe"',
        ") DO (",
        '    IF NOT DEFINED NVIDIA_SMI IF EXIST "%%~P" SET "NVIDIA_SMI=%%~P"',
        ")",
        "IF DEFINED NVIDIA_SMI GOTO :EOF",
        "",
        "FOR /F \"delims=\" %%I IN ('where.exe nvidia-smi.exe 2^>NUL') DO (",
        '    IF NOT DEFINED NVIDIA_SMI SET "NVIDIA_SMI=%%~fI"',
        ")",
        "GOTO :EOF",
    ]
    helper_text = newline.join(helper_lines) + newline

    file_text = "".join(lines)
    if "GOTO :EOF" not in file_text or ":detect_nvidia_smi" not in file_text:
        lines[endlocal_index:] = [helper_text]
        changed = True

    if not changed:
        return False

    write_text(path, "".join(lines))
    print(f"Updated {POST_INSTALL_BAT_PATH}")
    return True


def migrate(repo_root: Path, context: dict[str, Any]) -> None:
    _ = context

    changed_any = False
    changed_any = update_post_install_bat(repo_root) or changed_any

    if not changed_any:
        print("No repository changes were required for this migration.")
