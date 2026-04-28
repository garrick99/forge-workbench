"""Autonomous SM_120 hardware-mapping probe system.

Components:
  db          — SQLite store + content-addressed cubin/PTX cache
  generator   — PTX template engine
  runner      — probe pipeline (compile both, GPU run, decode, store)
  coverage    — coverage axes + bin synthesis
  scheduler   — autonomous probe loop
  miner       — SQL-based rule extraction
"""
from .db import ProbeDB
from .generator import ProbeSpec, materialize, expected_output, TEMPLATES
from .runner import run_probe
from .scheduler import probe_loop, seed_all_axes
from .miner import RULES, run_all as run_all_rules, print_summary as print_rule_summary
from .coverage import AXES

__all__ = (
    "ProbeDB", "ProbeSpec",
    "materialize", "expected_output", "TEMPLATES",
    "run_probe", "probe_loop", "seed_all_axes",
    "RULES", "run_all_rules", "print_rule_summary",
    "AXES",
)
