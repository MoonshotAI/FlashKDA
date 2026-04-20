#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sed "s|__REPO_ROOT__|${REPO_ROOT}|g" .clangd.template > .clangd
echo "Generated .clangd with REPO_ROOT=${REPO_ROOT}"

mkdir -p ~/.config/clangd
cp config.yaml ~/.config/clangd/
echo "Copied config.yaml to ~/.config/clangd/"
