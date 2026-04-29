"""Corvidae — Agent harness with modular plugin architecture."""

import warnings

# Suppress chromadb deprecation warning that triggers SIGINT in pytest async runner.
# This filter is applied at package load time, before any submodule imports occur.
warnings.filterwarnings("ignore", message=".*iscoroutinefunction.*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")
