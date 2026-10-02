#!/usr/bin/env bash
# T233 review probe: one sign of the rotated basis flipped in forward.js alone, against tests/forward-check.mjs on the made-up models
cd "$(dirname "$0")/.."
python -m pip install -q numpy pytest
python tests/t233_signs_probe.py
