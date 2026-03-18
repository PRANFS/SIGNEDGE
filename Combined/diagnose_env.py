from __future__ import annotations

import os
import platform
import shlex
import signal
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


CHECK_IMPORTS = [
    "numpy",
    "cv2",
    "PIL",
    "mediapipe",
    "tflite_runtime.interpreter",
    "sounddevice",
    "pyttsx3",
    "jax",
    "jaxlib",
    "moonshine_voice",
    "core.audio_stt",
    "core.audio_tts",
    "modes.startup_gate",
    "modes.asl_mode",
    "modes.eye_mode",
    "app",
]


KEY_PACKAGES = [
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "mediapipe",
    "tflite-runtime",
    "jax",
    "jaxlib",
    "moonshine-voice",
    "piper-tts",
    "sounddevice",
    "pillow",
]


def section(title: str) -> str:
    return f"\n{'=' * 20} {title} {'=' * 20}\n"


def run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def signal_name_from_returncode(return_code: int) -> str | None:
    if return_code >= 0:
        return None
    sig_number = -return_code
    try:
        return signal.Signals(sig_number).name
    except ValueError:
        return f"SIGNAL_{sig_number}"


def pretty_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def diagnose_import(module_name: str, cwd: Path) -> tuple[bool, str]:
    code = textwrap.dedent(
        f"""
        import faulthandler
        faulthandler.enable()
        import importlib
        importlib.import_module({module_name!r})
        print('OK')
        """
    )
    cmd = [sys.executable, "-X", "faulthandler", "-c", code]
    rc, out, err = run(cmd, cwd=cwd)
    sig = signal_name_from_returncode(rc)
    if rc == 0:
        return True, "import ok"
    if sig:
        return False, f"crashed with {sig} (return code {rc})"
    stderr_tail = err.strip().splitlines()[-1] if err.strip() else "no stderr"
    return False, f"failed with return code {rc}: {stderr_tail}"


def main() -> int:
    here = Path(__file__).resolve().parent
    report_path = here / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    lines: list[str] = []
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Working directory: {here}")
    lines.append(section("System"))
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"Python: {sys.version}")
    lines.append(f"Executable: {sys.executable}")

    lines.append(section("pip basics"))
    for pip_cmd in (
        [sys.executable, "-m", "pip", "--version"],
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "-m", "pip", "list", "--format=columns"],
    ):
        rc, out, err = run(pip_cmd, cwd=here)
        lines.append(f"$ {pretty_cmd(pip_cmd)}")
        lines.append(f"return code: {rc}")
        if out.strip():
            lines.append("stdout:")
            lines.append(out.rstrip())
        if err.strip():
            lines.append("stderr:")
            lines.append(err.rstrip())
        lines.append("")

    lines.append(section("Key package versions"))
    for pkg in KEY_PACKAGES:
        cmd = [sys.executable, "-m", "pip", "show", pkg]
        rc, out, err = run(cmd, cwd=here)
        if rc == 0 and out.strip():
            lines.append(out.rstrip())
            lines.append("")
        else:
            lines.append(f"{pkg}: not installed or not found")

    lines.append(section("Isolated import checks"))
    failed: list[tuple[str, str]] = []
    for module in CHECK_IMPORTS:
        ok, detail = diagnose_import(module, cwd=here)
        status = "PASS" if ok else "FAIL"
        lines.append(f"[{status}] {module}: {detail}")
        if not ok:
            failed.append((module, detail))

    lines.append(section("Quick hints"))
    machine = platform.machine().lower()
    if "aarch64" in machine or "arm" in machine:
        lines.append("ARM detected: wheels for jax/jaxlib/mediapipe can be unstable on Raspberry Pi.")
        lines.append("If import crashes on one of these, uninstall it first and retest app startup.")
    lines.append("If a module shows SIGBUS, focus on reinstalling only that package and its direct deps.")
    lines.append("Run with: python -X faulthandler app.py for native crash traces.")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Diagnostic report written to: {report_path}")
    if failed:
        print("\nImport failures detected:")
        for module, detail in failed:
            print(f" - {module}: {detail}")
        return 1

    print("\nAll isolated import checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
