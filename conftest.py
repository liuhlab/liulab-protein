"""Wiring, and nothing else: the suite's two guards, declared where they reach ``src/``.

A conftest reaches the directory it sits in and everything below, and nothing above — so
``tests/conftest.py`` reaches tests and only tests. The package's docstring examples are
collected from ``src/`` (``--doctest-modules``, see ``pyproject.toml``), which is outside that
tree, and a doctest item is a test: it runs behind both autouse guards or it is the hole in
them. This file is the one conftest above both trees, so declaring the guards here is what
gives them the reach the promise already claims. The guards themselves live in
:mod:`tests._guards`.
"""

pytest_plugins = ["tests._guards"]
