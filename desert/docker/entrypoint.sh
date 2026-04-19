#!/bin/sh
set -e
# Use `python -m cli.main` so Typer always runs. `uv run desert-cli worker` can be resolved
# incorrectly by uv (invoking worker argparse with argv still containing "worker").
mode="${DESERT_MODE:-orchestrator}"
case "$mode" in
  orchestrator)
    exec python -m cli.main orchestrator "$@"
    ;;
  worker)
    exec python -m cli.main worker "$@"
    ;;
  *)
    echo "DESERT_MODE must be 'orchestrator' or 'worker' (got: $mode)" >&2
    exit 1
    ;;
esac
