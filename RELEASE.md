# Release process

This document covers the one-time PyPI publisher setup plus the
per-release tag-and-push workflow.

## One-time setup: PyPI Trusted Publisher

`.github/workflows/release.yml` uses **PyPI's Trusted Publisher**
mechanism rather than an API token. Configure once:

1. Sign in to <https://pypi.org/> (create the `slack-certify-mapf`
   project if not already reserved).
2. **Project settings → Publishing → Add a new publisher** →
   **GitHub Actions**.
3. Fill in:
   - Owner: `anonymized`
   - Repository name: `slack-certify-mapf-test`
   - Workflow filename: `release.yml`
   - Environment name: *(leave blank)*
4. Save. PyPI now trusts that workflow to publish without an API
   token; no secrets are required on the GitHub side.

The same dance is required on TestPyPI if you want to publish a
release-candidate to <https://test.pypi.org/> first; update the
workflow's `repository-url` accordingly.

## Per-release workflow

1. Bump the version in `pyproject.toml` and `CITATION.cff`.
2. Move the `[Unreleased]` section of `CHANGELOG.md` into a new
   `[X.Y.Z] - YYYY-MM-DD` block; leave an empty `[Unreleased]` above
   it.
3. Commit on a `release/X.Y.Z` branch, open a PR, merge to `main`.
4. Tag and push:
   ```bash
   git checkout main && git pull
   git tag -s vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
5. `.github/workflows/release.yml` picks up the tag, builds the
   sdist + wheel, publishes to PyPI via Trusted Publisher, and
   creates a GitHub release whose body is extracted from the
   matching `## [X.Y.Z]` section of `CHANGELOG.md`.

## Verifying a release

```bash
pip install --upgrade slack-certify-mapf
python -c "from slackcertify import __version__; print(__version__)"
```

## Post-release badge swap

After the first successful PyPI publish and Zenodo deposit
creation, restore the dynamic badges in `README.md`:

1. PyPI version badge: replace

   ```markdown
   ![PyPI](https://img.shields.io/badge/pypi-pending%20release-lightgrey)
   ```

   with

   ```markdown
   [![PyPI](https://img.shields.io/pypi/v/slack-certify-mapf)](https://pypi.org/project/slack-certify-mapf/)
   ```

2. Python versions badge: replace

   ```markdown
   ![Python](https://img.shields.io/badge/python-3.10%2B-blue)
   ```

   with

   ```markdown
   ![Python](https://img.shields.io/pypi/pyversions/slack-certify-mapf)
   ```

3. DOI badge: replace

   ```markdown
   ![DOI](https://img.shields.io/badge/DOI-pending%20release-lightgrey)
   ```

   with

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.<ACTUAL_DEPOSIT_ID>.svg)](https://doi.org/10.5281/zenodo.<ACTUAL_DEPOSIT_ID>)
   ```

   using the deposit ID assigned by Zenodo on first archival.

4. Remove the "PyPI package and Zenodo DOI will be published
   with the first tagged release" sentence directly below the
   badge row.

5. Restore the BibTeX citation blocks in README.md's "Citing
   this work" section:

   ```bibtex
   @inproceedings{anon_slack_2026,
       author    = {Anonymous Author and Anonymous Co-author},
       title     = {Slack-Certified One-Shot MAPF: ...},
       booktitle = {Proceedings of ASYU},
       year      = {2026},
       doi       = {<ACTUAL_PAPER_DOI>}
   }

   @software{anon_slack_certify_mapf_2026,
       author    = {Anonymous Author},
       title     = {slack-certify-mapf: ...},
       year      = {2026},
       version   = {<RELEASE_VERSION>},
       doi       = {<ACTUAL_ZENODO_DOI>},
       url       = {<REPOSITORY_URL>}
   }
   ```

   Fill in `<ACTUAL_PAPER_DOI>`, `<ACTUAL_ZENODO_DOI>`,
   `<RELEASE_VERSION>`, and `<REPOSITORY_URL>` with real
   values. Update `CITATION.cff` to match.

Confirm by viewing the rendered `README.md` on GitHub.
