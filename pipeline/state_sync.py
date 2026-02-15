"""Automate regeneration of derivable sections in .claude/system/STATE.md.

Preserves human-written sections ("What's Working", "What's Not Working",
"Next Steps") while regenerating "Active src/ Files", "Filesystem", and
"OSC Environment" from the project DB and filesystem.

Usage:
    python -m pipeline.state_sync preview   # Print updated STATE.md without writing
    python -m pipeline.state_sync update    # Write updated STATE.md in-place
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path(".claude/system/STATE.md")

# Sections that are regenerated (all others are preserved verbatim)
_REGENERATED = {"Active `src/` Files", "Filesystem", "OSC Environment"}


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Split STATE.md into (header, body) pairs by ``## `` headers.

    The first pair has header="" for content before the first ``## ``.
    """
    parts: list[tuple[str, str]] = []
    current_header = ""
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            parts.append((current_header, "".join(current_lines)))
            current_header = line.strip().removeprefix("## ")
            current_lines = [line]
        else:
            current_lines.append(line)

    parts.append((current_header, "".join(current_lines)))
    return parts


def _db_counts() -> tuple[int, int, int]:
    """Return (datasets, runs, metrics) counts from project DB."""
    try:
        from .db import get_connection
        conn = get_connection()
        ds = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        metrics = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        conn.close()
        return ds, runs, metrics
    except Exception as e:
        log.warning("Could not read project DB: %s", e)
        return 0, 0, 0


def _inode_usage() -> str:
    """Get inode usage for home directory."""
    try:
        result = subprocess.run(
            ["df", "-i", str(Path.home())],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5:
                used = parts[2]
                total = parts[3] if parts[3] != "-" else parts[1]
                return f"{used} / {total}"
    except Exception as e:
        log.warning("Could not get inode usage: %s", e)
    return "unknown"


def _active_src_files() -> str:
    """Generate the Active src/ Files section."""
    lines = [
        "## Active `src/` Files\n",
        "\n",
        "Essential (imported by pipeline):\n",
    ]
    src = Path("src")
    if src.exists():
        for subdir in ["models", "preprocessing", "training"]:
            p = src / subdir
            if p.exists():
                py_files = sorted(f.name for f in p.glob("*.py") if f.name != "__init__.py")
                if py_files:
                    lines.append(f"- `src/{subdir}/` — {', '.join(py_files)}\n")
    lines.append("\n")
    return "".join(lines)


def _filesystem_section() -> str:
    """Generate the Filesystem section."""
    inode = _inode_usage()
    lines = [
        "## Filesystem\n",
        "\n",
        f"- **Inode usage**: {inode}\n",
    ]
    # Check for conda envs
    conda_envs = Path.home() / ".conda" / "envs"
    if conda_envs.exists():
        envs = sorted(d.name for d in conda_envs.iterdir() if d.is_dir())
        if envs:
            lines.append(f"- **Conda envs**: {', '.join(envs)}\n")
    lines.append("\n")
    return "".join(lines)


def _osc_environment_section() -> str:
    """Generate the OSC Environment section."""
    lines = [
        "## OSC Environment\n",
        "\n",
        "- **Home**: `/users/PAS2022/rf15/` (NFS, permanent)\n",
        "- **Scratch**: `/fs/scratch/PAS1266/` (GPFS, 90-day purge)\n",
        "- **Snakemake cache**: `/fs/scratch/PAS1266/snakemake-cache/`\n",
        "- **Project DB**: `data/project.db` (SQLite — datasets, runs, metrics, epoch_metrics)\n",
        "- **Dashboard**: `docs/dashboard/` (GitHub Pages — static JSON + D3.js)\n",
        "- **Conda**: `module load miniconda3/24.1.2-py310 && conda activate gnn-experiments`\n",
    ]
    return "".join(lines)


def _update_date_line(preamble: str) -> str:
    """Update the **Date**: line in the preamble."""
    today = date.today().isoformat()
    return re.sub(
        r"\*\*Date\*\*:\s*\d{4}-\d{2}-\d{2}",
        f"**Date**: {today}",
        preamble,
    )


def _update_db_counts(text: str) -> str:
    """Update project DB counts in the Data Management subsection."""
    ds, runs, metrics = _db_counts()
    if ds == 0 and runs == 0:
        return text
    return re.sub(
        r"(\*\*Project DB\*\*:.*?—\s*)\d+ datasets,\s*\d+ runs,\s*\d+ metrics",
        rf"\g<1>{ds} datasets, {runs} runs, {metrics} metrics",
        text,
    )


def update_state(preview: bool = False) -> str:
    """Regenerate derivable sections of STATE.md.

    Args:
        preview: If True, return the updated content without writing to disk.

    Returns:
        The full updated STATE.md content.
    """
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"{STATE_PATH} not found")

    original = STATE_PATH.read_text()
    sections = _parse_sections(original)

    generators = {
        "Active `src/` Files": _active_src_files,
        "Filesystem": _filesystem_section,
        "OSC Environment": _osc_environment_section,
    }

    result_parts: list[str] = []
    for header, body in sections:
        if header == "":
            # Preamble — update date
            result_parts.append(_update_date_line(body))
        elif header in generators:
            result_parts.append(generators[header]())
        else:
            # Preserve human-written sections, but update DB counts if present
            result_parts.append(_update_db_counts(body))

    content = "".join(result_parts)

    if not preview:
        STATE_PATH.write_text(content)
        log.info("Updated %s", STATE_PATH)

    return content


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pipeline.state_sync",
        description="Regenerate derivable sections of STATE.md",
    )
    parser.add_argument(
        "action", choices=["preview", "update"],
        help="'preview' prints without writing, 'update' writes in-place",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    )

    content = update_state(preview=(args.action == "preview"))
    if args.action == "preview":
        print(content)


if __name__ == "__main__":
    main()
