---
search:
  exclude: true
---

# Domain docs

**Single-context**: one `CONTEXT.md` and one `docs/adr/` at the repo root.

Before exploring the code, read `CONTEXT.md` for vocabulary and any ADR touching the area
you are about to change. If either is missing, proceed silently — `/domain-modeling` creates
them lazily, when a term or a decision is actually resolved.

Use the glossary's words. When your output names a domain concept — an issue title, a test
name, a refactor proposal — use the term as `CONTEXT.md` defines it, not a synonym it lists
as avoided. A concept defined nowhere is a signal: usually language the project does not
use, occasionally a real gap.

If your output contradicts an ADR, say so rather than silently overriding it.

`CONTEXT.md` is a glossary and nothing else — no implementation detail, no spec, no scratch
notes. An ADR is written only when a decision is hard to reverse, surprising without
context, and the result of a real trade-off. All three, or no record.

Both are agent-facing and capped — see `docs/agents/writing.md`.
