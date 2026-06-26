"""Centralized filesystem paths for the RAG project (single source of truth).

Modules import these instead of recomputing the repo root from ``__file__`` so
that moving a module between sub-packages does not change which directory it
resolves to.
"""
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parent.parent  # rag/paths.py -> RAG/
WORKSPACE_DIR = RAG_ROOT.parent                     # Intership_U/
OUTPUTS_DIR = RAG_ROOT / "outputs"
EVAL_DIR = RAG_ROOT / "eval"
RULES_DIR = RAG_ROOT / "rules"
# The web frontend lives inside the repo at RAG/frontend (built UI under dist/).
FRONTEND_DIR = RAG_ROOT / "frontend" / "dist"
