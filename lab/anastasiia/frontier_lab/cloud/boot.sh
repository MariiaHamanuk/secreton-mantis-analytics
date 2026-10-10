#!/bin/bash
set -e
cd ~
sudo apt-get update -qq >/dev/null 2>&1 || true
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
rm -rf repo && tar xzf repo.tgz
cd repo && ~/.local/bin/uv sync --frozen >/dev/null 2>&1
mkdir -p outputs/logs
~/.local/bin/uv run python -c "import shockbench_flow, scipy, numpy; print('env ok', numpy.__version__, scipy.__version__)"
nproc; free -g | sed -n 2p
