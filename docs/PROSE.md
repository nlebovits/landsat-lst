# Prose style

Write documentation for a technical reader who may be in a hurry. Lead with
the result, then explain the mechanism and limits that help the reader use or
judge it.

Use clear, direct, and conversational language. Prefer concrete verbs and name
the component that performs an action. Address the reader as "you" in
instructions. Keep technical terms when they are more precise than a plain
substitute.

Use sentence-case headings. Keep examples executable and claims testable.
State uncertainty, costs, and unsupported cases without promotional language.

## Avoid formulaic prose

These rules apply to every contributor, including an LLM that drafts or edits
documentation. Before committing generated prose, remove conversational
residue, unsupported enthusiasm, and repeated sentence frames.

Vary sentence length and structure. Prefer syntax that states the relationship
between ideas over syntax that merely adds one item after another.

Avoid enumerative parataxis. It often appears as:

- repeated inline lists;
- compound predicates;
- balanced coordinate clauses; or
- affirmative-negative sentence pairs.

Common frames include `It does A, B, and C`, `X does A and does B`, and
`It does A. It does not do B`. Use subordination when ideas have a causal,
conditional, temporal, concessive, or purposive relationship. Reserve inline
lists for items that form a meaningful set.

Replace the resultative frame `X, so you can Y` with the relationship that
makes Y possible:

```text
Because the repository includes the fixtures, you can reproduce the check.
```

Do not repeat a grammatical frame across adjacent sentences or paragraphs.
Cut stock transitions, filler, chatbot closings, dramatic setup clauses, and
contrasts that turn two facts into a slogan. Limit em dashes rather than using
an em dash as a default transition. End on the final substantive point instead
of a summary that repeats it.

The rules identify text patterns, not authorship. A person can write formulaic
prose, and an LLM can write clean prose.

## Automated checks

[Vale](https://vale.sh/) parses Markdown before applying the repository-owned
rules in `styles/`. The `Landsat-*` styles cover project terminology,
mechanics, and formulaic voice patterns. The pinned Microsoft package supplies
the broader developer-documentation rules. Checks that conflict with local
conventions or duplicate a local rule are disabled in `.vale.ini`.

The Readability package reports the Automated Readability Index (ARI) and
Flesch Reading Ease for each document. Both findings are suggestions, which
makes them visible in the advisory audit without blocking a commit. Use the
scores to compare a document before and after an edit. Technical names can
keep sound documentation outside the package's general targets.

Improve readability by unpacking abstractions and making relationships clear.
Do not shorten sentences or replace precise terms only to improve a score.

[proselint](https://github.com/amperser/proselint) reports selected clichés,
hedges, redundant phrases, mixed metaphors, and commercial language. Its
findings are advisory because they require editorial judgment.

Run the blocking commit-stage checks with the same command used locally:

```bash
uv run prek run --all-files --show-diff-on-failure
```

Run the advisory checks through their pinned hook environments:

```bash
uv run prek run vale-audit --all-files --stage manual
uv run prek run proselint --all-files --stage manual
```

The normal Vale hook and the dedicated Prose workflow block error-level
findings in handwritten Markdown. They exclude generated `AGENTS.md` context,
retained `docs/evidence/` bundles, and `results/` artifacts.

The first repository-wide audit found older title-case headings, hype words,
and repeated sentence frames throughout historical ADRs and findings. Those
checks remain warnings until maintainers edit the affected documents. Clear
mechanical errors and unmistakable chatbot residue block immediately. This
calibration keeps the gate green without rewriting the scientific record as
part of a tooling change.

## Suppress a false positive

Prefer a narrow suppression and explain why the prose must keep its form.

```markdown
<!-- vale Landsat-Mechanics.Headings = NO -->
## An External Name That Uses Title Case
<!-- vale Landsat-Mechanics.Headings = YES -->
```

Use `<!-- vale off -->` only for a whole block that Vale cannot parse usefully.
Do not suppress a finding only to make the check pass.

## Rule ownership

The repository owns the `Landsat-*` rules and their tests. They are adapted
from the Portolan prose rules at the commit named in `styles/NOTICE`. `vale
sync` downloads the pinned packages into ignored directories.

Each custom rule has one failing and one passing example in
`tests/test_prose_styles.py`. Add both examples when you add or change a rule.
