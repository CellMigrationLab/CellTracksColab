"""TLS-resilient JupyterLab launcher for CellTracksColab.

Certificate handling is configured before JupyterLab imports Tornado. On
Windows this prevents one malformed certificate-store entry from terminating
the application. Certificate and hostname verification remain enabled in every
supported mode.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import ssl
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_DISPLAY_NAME = "CellTracksColab"
SCRIPT_DIR = Path(__file__).resolve().parent
_INSTALLED_LAYOUT = (SCRIPT_DIR / "notebooks").is_dir()
if _INSTALLED_LAYOUT:
    APP_ROOT = SCRIPT_DIR
    PREFIX = APP_ROOT.parent
else:
    # Repository layout: app/python_scripts/launch_jupyter.py
    repository_root = SCRIPT_DIR.parents[1]
    APP_ROOT = repository_root if (repository_root / "notebooks").is_dir() else SCRIPT_DIR
    PREFIX = APP_ROOT
DEFAULT_NOTEBOOK = APP_ROOT / "notebooks" / "Welcome.ipynb"
LOG_PATH = PREFIX / "launcher_debug.log"


def _write_log(message: str, *, console: bool = True) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    if console:
        print(message, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        # Logging must never prevent the application from launching.
        pass


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _project_ca_bundle_environment_variable() -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", PROJECT_DISPLAY_NAME).strip("_").upper()
    return f"{normalized or 'LABCONSTRICTOR_APP'}_CA_BUNDLE"


def _probe_ssl() -> None:
    """Create a verified client context without making a network request."""
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    if context.verify_mode != ssl.CERT_REQUIRED:
        raise RuntimeError("TLS certificate verification is unexpectedly disabled.")
    if not context.check_hostname:
        raise RuntimeError("TLS hostname verification is unexpectedly disabled.")


def _inject_truststore() -> None:
    """Route Python SSL contexts through the operating-system trust API."""
    import truststore

    truststore.inject_into_ssl()


def _select_ca_bundle() -> Path:
    """Choose an explicit CA bundle for verified installer and runtime fallback."""
    for variable in (
        _project_ca_bundle_environment_variable(),
        "LABCONSTRICTOR_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        candidate = os.environ.get(variable, "").strip()
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path.resolve()

    import certifi

    return Path(certifi.where()).resolve()


def _install_bundle_fallback(bundle: Path) -> None:
    """Use a verified PEM bundle without enumerating the Windows cert store."""
    original_context_type = ssl.SSLContext

    def create_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | os.PathLike[str] | None = None,
        capath: str | os.PathLike[str] | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        if purpose == ssl.Purpose.SERVER_AUTH:
            context = original_context_type(ssl.PROTOCOL_TLS_CLIENT)
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
        elif purpose == ssl.Purpose.CLIENT_AUTH:
            context = original_context_type(ssl.PROTOCOL_TLS_SERVER)
        else:
            raise ValueError(purpose)

        if cafile or capath or cadata:
            context.load_verify_locations(
                cafile=os.fspath(cafile) if cafile else None,
                capath=os.fspath(capath) if capath else None,
                cadata=cadata,
            )
        else:
            context.load_verify_locations(cafile=str(bundle))

        keylogfile = os.environ.get("SSLKEYLOGFILE")
        if keylogfile and hasattr(context, "keylog_filename"):
            context.keylog_filename = keylogfile
        return context

    ssl.create_default_context = create_context  # type: ignore[assignment]
    ssl._create_default_https_context = create_context  # type: ignore[attr-defined]
    os.environ["SSL_CERT_FILE"] = str(bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))
    os.environ.setdefault("CURL_CA_BUNDLE", str(bundle))


def configure_tls() -> tuple[str, list[str]]:
    """Prepare TLS before Jupyter/Tornado imports and return mode plus notes."""
    notes: list[str] = []

    try:
        if "truststore" not in getattr(ssl.SSLContext, "__module__", ""):
            _inject_truststore()
        _probe_ssl()
        return "native truststore", notes
    except Exception as exc:
        notes.append(f"Native truststore activation failed: {exc}")

    try:
        bundle = _select_ca_bundle()
        _install_bundle_fallback(bundle)
        _probe_ssl()
        notes.append(f"Using verified CA bundle fallback: {bundle}")
        return "verified CA bundle fallback", notes
    except Exception as exc:
        notes.append(f"CA bundle fallback failed: {exc}")
        raise RuntimeError("Unable to create a verified TLS context. " + " | ".join(notes)) from exc


def _diagnostics(tls_mode: str, notes: list[str]) -> dict[str, object]:
    return {
        "status": "ok",
        "application": PROJECT_DISPLAY_NAME,
        "tls_mode": tls_mode,
        "notes": notes,
        "ca_bundle_override_variable": _project_ca_bundle_environment_variable(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "openssl": ssl.OPENSSL_VERSION,
        "platform": platform.platform(),
        "jupyterlab": _package_version("jupyterlab"),
        "jupyter_server": _package_version("jupyter-server"),
        "tornado": _package_version("tornado"),
        "truststore": _package_version("truststore"),
        "pip_system_certs": _package_version("pip-system-certs"),
        "certifi": _package_version("certifi"),
        "prefix": str(PREFIX),
        "log": str(LOG_PATH),
    }


def _show_error_dialog(message: str) -> None:
    if os.name != "nt":
        return
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(f"{PROJECT_DISPLAY_NAME} could not start", message)
        root.destroy()
    except Exception:
        pass


def run_jupyter(notebook: Path) -> int:
    if not notebook.is_file():
        raise FileNotFoundError(f"Welcome notebook not found: {notebook}")

    # Import only after configure_tls() has completed. Tornado creates SSL
    # contexts during import on some versions.
    from jupyterlab.labapp import main

    home = Path.home().resolve()
    try:
        notebook.resolve().relative_to(home)
        root_dir = home
    except ValueError:
        # All-users installations can live outside the home directory.
        root_dir = PREFIX.resolve()

    sys.argv = [
        "jupyter-lab",
        str(notebook.resolve()),
        "--ServerApp.ip=127.0.0.1",
        f"--ServerApp.root_dir={root_dir}",
        "--ServerApp.shutdown_no_activity_timeout=30",
        "--FileContentsManager.always_delete_dir=True",
    ]
    result = main()
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Launch {PROJECT_DISPLAY_NAME} in its packaged JupyterLab environment safely."
    )
    parser.add_argument("notebook", nargs="?", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate TLS and Jupyter imports without starting the server.",
    )
    parser.add_argument(
        "--print-ca-bundle",
        action="store_true",
        help="Print the selected verified CA bundle path and exit.",
    )
    args = parser.parse_args(argv)

    try:
        if args.print_ca_bundle:
            print(_select_ca_bundle(), flush=True)
            return 0

        _write_log("Starting application launcher")
        tls_mode, notes = configure_tls()
        _write_log(f"TLS mode: {tls_mode}")
        for note in notes:
            # Recovery details remain available in launcher_debug.log and
            # self-test JSON without alarming users during a normal launch.
            _write_log(note, console=False)

        import jupyterlab.labapp  # noqa: F401
        import tornado.netutil  # noqa: F401

        diagnostics = _diagnostics(tls_mode, notes)
        if args.self_test:
            print(json.dumps(diagnostics, indent=2), flush=True)
            _write_log("Launcher self-test completed successfully")
            return 0

        _write_log(f"Opening notebook: {args.notebook}")
        return run_jupyter(args.notebook)
    except KeyboardInterrupt:
        _write_log("Launcher interrupted by user")
        return 130
    except Exception as exc:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _write_log(details.rstrip())
        friendly = (
            f"{PROJECT_DISPLAY_NAME} could not start, but no certificate-store changes are required.\n\n"
            f"A diagnostic log was written to:\n{LOG_PATH}\n\n"
            f"Technical message: {exc}"
        )
        print("\n" + friendly, file=sys.stderr, flush=True)
        _show_error_dialog(friendly)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
