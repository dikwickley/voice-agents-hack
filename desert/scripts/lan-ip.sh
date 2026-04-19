#!/usr/bin/env bash
# Print a reachable host address for DESERT_P2P_ANNOUNCE_ADDR.
# Order of preference:
#   1. $DESERT_LAN_IP override (if set and non-empty).
#   2. Linux: route to 8.8.8.8.
#   3. macOS: en0 then en1.
#   4. Fallback: host.docker.internal (works from any Docker Desktop container).
set -euo pipefail

if [[ -n "${DESERT_LAN_IP:-}" ]]; then
  printf '%s\n' "${DESERT_LAN_IP}"
  exit 0
fi

LAN_IP=""

if command -v ip >/dev/null 2>&1; then
  LAN_IP="$(ip -4 route get 8.8.8.8 2>/dev/null \
    | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}')"
fi

if [[ -z "${LAN_IP}" ]] && command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
  [[ -n "${LAN_IP}" ]] || LAN_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi

# Never print empty: callers embed this in `/ip4/${LAN_IP}/tcp/4001`, and an
# empty value produces an invalid multiaddr that crashes every worker.
printf '%s\n' "${LAN_IP:-host.docker.internal}"
