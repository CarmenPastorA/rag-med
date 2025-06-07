"""Basic tests for the RAG orchestrator module."""

import importlib
import inspect


def test_run_rag_pipeline_exists():
    """Ensure run_rag_pipeline is exposed as a callable."""
    module = importlib.import_module("tools.rag_orchestrator")
    assert hasattr(module, "run_rag_pipeline")
    assert inspect.isfunction(module.run_rag_pipeline)
