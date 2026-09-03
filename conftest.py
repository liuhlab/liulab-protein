"""Wiring, and nothing else: the suite's two guards, declared where they reach ``src/``.

A conftest reaches the directory it sits in and below, and nothing above, so
``tests/conftest.py`` would not reach the docstring examples collected from ``src/`` — and a
doctest item is a test, which runs behind both autouse guards or is the hole in them. This
file is the one conftest above both trees. The guards live in :mod:`tests._guards`.
"""

pytest_plugins = ["tests._guards"]
