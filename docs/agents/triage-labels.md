---
search:
  exclude: true
---

# Triage labels

Each label string equals its role name — the mapping is the identity, so there is no
translation step to get wrong. Apply with `gh issue edit <n> --add-label "..."`.

| Label | Meaning |
| --- | --- |
| `needs-triage` | Maintainer needs to evaluate this issue |
| `needs-info` | Waiting on reporter for more information |
| `ready-for-agent` | Fully specified, ready for an AFK agent |
| `ready-for-human` | Requires human implementation |
| `wontfix` | Will not be actioned |

`/triage` is for issues you did not create. Tickets `/to-tickets` emits are already
agent-ready and skip triage.
