#!/usr/bin/env bash
# Run local CodeSpec2 through Pier without exposing hidden DeepSWE materials.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ -f .env.local ]]; then
  # shellcheck disable=SC1091
  source .env.local
fi

export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

# The custom adapter also reads .env.local in the host Pier process.  Sourcing
# it here additionally supports its normal Pier configuration/egress handling.
# The caller supplies only normal `pier run` arguments, for example:
# bash run_codespec2_pier.sh -p tasks/gql-incremental-graphql-delivery \
#   --model deepseek/deepseek-v4-pro
exec pier run \
  --agent-import-path codespec2_pier.agent:CodeSpec2PierAgent \
  "$@"
