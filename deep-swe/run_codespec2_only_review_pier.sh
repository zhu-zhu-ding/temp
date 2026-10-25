#!/usr/bin/env bash
# Re-review the fixed prior CodeSpec2 attempt without exposing its old checkout.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ -f .env.local ]]; then
  # shellcheck disable=SC1091
  source .env.local
fi

export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

exec pier run \
  --agent-import-path codespec2_pier.agent:CodeSpec2OnlyReviewPierAgent \
  "$@"
