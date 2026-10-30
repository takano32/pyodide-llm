"""tests/cpu.py (T.494): the CPU a tool runs on, for its log: GitHub's runners are of several CPUs, and an only_extra
run says nothing of its runner but what its tool says (AGENTS.md: read speeds CPU by CPU)."""
from pathlib import Path


def cpu_name():
    """the CPU's name as Linux says it (x86-64's model name; arm64 says none: unknown), for runs to be read CPU by CPU"""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"
