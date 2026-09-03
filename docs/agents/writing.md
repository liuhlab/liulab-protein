---
search:
  exclude: true
---

# Writing rules

Three rules. Two are checked by `vale` in `pixi run check`; the third is on you.

## 1. Be concise

Shorter beats longer — in documents, issues, commit messages, and replies. If a sentence
survives deletion without loss, delete it. The caps below are ceilings, not targets.

## 2. Agent-facing documents have word caps

| Files | Cap |
| --- | --- |
| `AGENTS.md`, `CONTEXT.md`, `docs/agents/*`, `skills/*/SKILL.md` | 1000 (`Lab.LengthDoc`) |
| `docs/adr/*` | 400 (`Lab.LengthAdr`) |
| `docs/research/*` | none — research notes are long by nature |

`CONTEXT.md` is capped a second way: **200 words per glossary entry**, checked by
`conformance`, not by vale. A glossary grows one term at a time, so the file is the wrong
unit to measure.

**Measure with the gate, not with `wc`.** `wc -w` counts table pipes, link targets and
shell flags as words; vale does not, and vale is what enforces the cap. The gap scales with
markup — measured across this repo it runs from 10% on prose to 65% on a page that is mostly
a table, so `wc` will tell you a reference page is near a cap it is nowhere near. Run
`pixi run vale` and believe it.

The caps are dials with tight defaults. Raising one is a one-line diff in `styles/Lab/` —
do that deliberately, and say why in the commit message. Do not raise a cap because a
document ran long; that is the cap working.

## 3. Human-facing prose avoids jargon and stays readable

`README.md` and every `docs/` page a human browses: no terms from the lab jargon list
(`Lab.Jargon` — architecture-speak and Latinate verbs), and reading grade 11 or below
(`Lab.Readability`). No length cap — a tutorial is as long as the task.

**When `Lab.Readability` fails, match a passing exemplar — do not attack the number.** The
grade is a ratio over structure, so shaving words inside sentences you already committed to
moves it by hundredths. Find text that passes — an older section, a sibling page — and rewrite
toward how it is segmented. Measured on a 21,000-word changelog: optimising the grade directly
moved it 12.46 to 12.41; matching a passing section in the same file moved it under 11, while
cutting only 3% of the words and *raising* the entry count.

**This rule also covers what you say, not only what you write.** Nothing checks a chat
reply, so it is on you: when a human asks, answer in plain language and explain the term
you would otherwise reach for. An unexplained term in conversation is the same failure as
one in the README.

## Which is which

Agent-facing means written for a machine that has to act: capped, exempt from the jargon
and readability rules, kept out of the site navigation. Human-facing means written for a
person reading the published site: checked for jargon and readability, uncapped. A file is
one or the other — if you are adding a document and cannot tell, it is human-facing.
