"""What is specific to ESM: the backbone, and whatever else reads its weights.

A directory and not a flat ``esm.py``, because the lane above keeps its verb name — ESM-C is
one embedding provider of several to come, so what belongs to ESM groups here while
:mod:`protein.embed.embedding` stays at the lane root, provider-neutral.

**Nothing is re-exported here**, unlike the other packages in this tree.
:mod:`protein.embed` is the one import surface, so a name has one public path and not two.
"""
