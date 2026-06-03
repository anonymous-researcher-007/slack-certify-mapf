<!-- PR template for slack-certify-mapf. Fill in each section; delete
the comments before submitting. Use closing keywords like `Fixes #123`
in the description if the PR closes an issue. -->

## Summary

<!-- One-paragraph description of what this PR changes and why. -->

## Changes

<!-- Bullet list of concrete changes. Group by area
(certifier / experiments / analysis / docs / CI). -->

-
-

## Testing checklist

- [ ] `ruff check .` is clean
- [ ] `mypy src/slackcertify` reports no *new* errors
- [ ] `pytest -m "not slow and not integration"` passes locally
- [ ] If touching experiment runners: relevant smoke under
      `tests/integration/test_rq*_smoke.py` still passes
- [ ] If touching analysis: `python analysis/make_figures.py --smoke`
      still produces every PDF + tex
- [ ] If touching docs/CI: rendered the affected files locally to
      catch syntax errors

## Paper impact

<!-- Does this change affect any §V numbers, figure, or table? If yes,
list which RQ(s) are affected and whether the change requires re-running
the full sweep (scripts/repro_paper.sh) before camera-ready. -->

- [ ] No paper impact
- [ ] Cosmetic / docs only
- [ ] Changes a figure or table (specify which)
- [ ] Changes a reported §V number (must re-run scripts/repro_paper.sh)

## Breaking changes

<!-- Public API, CLI flags, or CSV schema changes? If yes, note the
deprecation strategy. -->

- [ ] None
- [ ] Public API change (described above)
- [ ] CLI / runner config change (described above)
- [ ] CSV / manifest schema change (described above)
