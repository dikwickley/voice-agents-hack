#!/bin/sh
set -e
# Use `python -m cli.main` so Typer always receives the right argv in all environments.
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
