# Test fixtures

Small, subsampled **real** files — never a large one, and never a made-up one where a real one
would do. Nothing in the suite reaches the network (`tests/_guards.py`), so anything a test
needs to read lives here.

**There are none yet.** `seq.py` is pure, so its tests need no bytes at all.

Every fixture added here gets a row in a table on this page, in the same commit as the file:
what it is, the URL or command its bytes came from, and every way it departs from the source.
A fixture whose provenance is not written down is one nobody can check later, and being
checkable is the whole reason to prefer a real file to an invented one.
