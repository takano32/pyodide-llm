#!/usr/bin/env bash
# T262: apt's time-outs on a runner of GitHub, before anything that runs apt there (`npx playwright-core install
# --with-deps`, `sudo apt-get install`). Run 36902097346 of main (gpu-prompt.yml) had its Chromium job stay in the apt of
# that install (the last line of the log was a "Get:" of archive.ubuntu.com's noble-security InRelease) until the job's
# 120 minutes were over: the tests never began. With this, a download that stops is given up after 30 seconds and tried
# 3 times more. The other half of the protection is the step's own `timeout-minutes: 20` in the workflow (a step's limit
# cannot be set from a script, nor from the steps of a composite action).
#
# Linux runners only: Windows and macOS have no apt (the step's limit is what they get), and a development machine's
# /etc is not this script's to write.
set -euo pipefail
[ "${GITHUB_ACTIONS:-}" = true ] && [ "$(uname -s)" = Linux ] || exit 0
printf 'Acquire::http::Timeout "30";\nAcquire::https::Timeout "30";\nAcquire::Retries "3";\n' | sudo tee /etc/apt/apt.conf.d/99-ci-timeouts > /dev/null
