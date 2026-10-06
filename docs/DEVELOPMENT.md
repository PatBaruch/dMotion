# Development

The [README](../README.md) is the complete user guide. The [file map](FILE_LAYOUT.md)
identifies modules and local artifacts. This branch maintains YOLO26m only.

## Core environment and checks

Use Python 3.11–3.13 and the dependency versions in `uv.lock`:

```sh
uv sync --locked --python 3.11
make setup-workflow
make check
make audit
make package-check
```

If `make setup` already installed uv, use `.tools/bin/uv`; put `.tools/bin` on PATH
for `make package-check`. Core checks need no camera, audio device, model download,
or vision packages. Use `make setup` for a separate hardware/model environment.
Do not replace an environment used by an active training run with core-only sync.

Use Context7 for uncertain external behavior and verify against the pinned source.
Use codebase-memory-mcp for architecture/call tracing, select the exact checkout,
check coverage/freshness, and read source before editing. Graph omissions do not
prove a feature is absent.

## Model and data contracts

Inference loads a user-selected single-class cash checkpoint through `YOLO`.
Accept `cash` and legacy `money_spread` labels. It must not silently download a
generic model when a cash checkpoint is missing. Old mode aliases preserve model
paths and confidence. Training initializes YOLO26m and saves a separate candidate;
it must not replace the user's live checkpoint or select confidence from test data.

Reviewed truth lives in dataset manifests. AI suggestions stay pending, including
empty detections. Recording groups remain intact across splits; related videos
from a physical session must share a group. Keep historical suggestion metadata
readable even after the generating backend is retired. Existing datasets remain
compatible; the legacy exported class name is retained.

Unit tests establish software behavior. Hardware acceptance and held-out model
evaluation are separate checks. Follow [AI reliability](AI_RELIABILITY.md).

## Dependencies and publication

Change declarations and lock together; do not upgrade unrelated packages during
cleanup. YOLO26m needs no project CLIP or Transformers optional integration.

```sh
uv add --optional vision package-name
uv lock --upgrade-package package-name
make check
```

The remote is [PatBaruch/dMotion](https://github.com/PatBaruch/dMotion). Start features
from current `origin/develop` in an isolated `feature/<name>` worktree. Keep model
binaries, datasets, outputs, credentials and unrelated work out of commits.
See [Git workflow](GIT_WORKFLOW.md) for checks, automatic PR completion and protected
merge rules. Pending or missing latest-commit review never counts as passed.

No publication license has been selected. Dependencies and external data retain
their own terms; assess them before distributing a product.
