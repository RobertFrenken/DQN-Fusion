"""Layer boundary enforcement tests.

Verifies the 3-layer import hierarchy under graphids/:
    config/    (top)     — never imports from pipeline/ or core/
    pipeline/  (middle)  — never has top-level imports from core/
    core/      (bottom)  — never imports from pipeline/

Uses AST analysis (no runtime imports needed).

Run:  python -m pytest tests/test_layer_boundaries.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "graphids"

CONFIG_DIR = PACKAGE_ROOT / "config"
PIPELINE_DIR = PACKAGE_ROOT / "pipeline"
CORE_DIR = PACKAGE_ROOT / "core"


def _collect_python_files(directory: Path) -> list[Path]:
    """Collect all .py files in a directory (recursively)."""
    return sorted(directory.rglob("*.py"))


def _extract_imports(filepath: Path) -> list[tuple[str, bool]]:
    """Extract import targets from a Python file.

    Returns list of (module_name, is_top_level) tuples.
    is_top_level is True if the import is at module scope (not inside a
    function, class, or if TYPE_CHECKING block).
    """
    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Determine if this import is at module scope
            # We consider it top-level if it's a direct child of the Module node
            top_level = _is_top_level_import(tree, node)

            if isinstance(node, ast.Import):
                for alias in node.names:
                    results.append((alias.name, top_level))
            elif node.module:
                results.append((node.module, top_level))
    return results


def _is_top_level_import(tree: ast.Module, target_node: ast.AST) -> bool:
    """Check if an import node is at module scope (not inside function/class/TYPE_CHECKING)."""
    for node in ast.iter_child_nodes(tree):
        if node is target_node:
            return True
        # Check if it's inside an `if TYPE_CHECKING:` block
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                if _node_contains(node, target_node):
                    return False
            elif isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                if _node_contains(node, target_node):
                    return False
    return False


def _node_contains(parent: ast.AST, target: ast.AST) -> bool:
    """Check if target node exists anywhere inside parent."""
    for child in ast.walk(parent):
        if child is target:
            return True
    return False


def _subpackage_imported(filepath: Path, top_level_only: bool = False) -> set[str]:
    """Return the set of graphids subpackage names imported by a file.

    Extracts the second-level module name from graphids.X imports.
    E.g. 'from graphids.config import ...' -> {'config'}
         'from graphids.core.models import ...' -> {'core'}
    """
    imports = _extract_imports(filepath)
    modules = set()
    for mod, is_top_level in imports:
        if top_level_only and not is_top_level:
            continue
        parts = mod.split(".")
        if len(parts) >= 2 and parts[0] == "graphids":
            modules.add(parts[1])  # "config", "pipeline", "core"
    return modules


class TestConfigLayerBoundary:
    """config/ must never import from pipeline/ or core/."""

    def test_config_no_pipeline_imports(self):
        violations = []
        for f in _collect_python_files(CONFIG_DIR):
            mods = _subpackage_imported(f)
            if "pipeline" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"config/ imports from pipeline/ (violates layer boundary):\n  "
            + "\n  ".join(violations)
        )

    def test_config_no_core_imports(self):
        violations = []
        for f in _collect_python_files(CONFIG_DIR):
            mods = _subpackage_imported(f)
            if "core" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"config/ imports from core/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )


class TestCoreLayerBoundary:
    """core/ must never import from pipeline/."""

    def test_core_no_pipeline_imports(self):
        violations = []
        for f in _collect_python_files(CORE_DIR):
            mods = _subpackage_imported(f)
            if "pipeline" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"core/ imports from pipeline/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )


class TestPipelineLayerBoundary:
    """pipeline/ must not have top-level imports from core/ (lazy/function-local OK)."""

    def test_pipeline_no_toplevel_core_imports(self):
        violations = []
        for f in _collect_python_files(PIPELINE_DIR):
            mods = _subpackage_imported(f, top_level_only=True)
            if "core" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"pipeline/ has top-level imports from core/ (should be lazy/function-local):\n  "
            + "\n  ".join(violations)
        )
