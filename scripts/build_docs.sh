#!/usr/bin/env bash
# Build the docs site (used by Vercel and locally).
#
# mkdocs serves files from `docs/`. The example notebooks live in
# `examples/` (so they're discoverable in the repo root). Copy them into
# `docs/examples/` at build time so mkdocs-jupyter can render them — same
# notebook, one place to edit, no symlinks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p docs/examples
cp examples/*.ipynb docs/examples/

python -m mkdocs build "$@"
