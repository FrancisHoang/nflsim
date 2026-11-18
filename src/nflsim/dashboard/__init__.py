"""Streamlit dashboard (optional extra: ``uv sync --extra dashboard``; ``nflsim dashboard``).

The pages call the same functions as the CLI (``nflsim.weekly.run_week``, ...), so the
dashboard shows the same numbers as ``nflsim week`` for the same seed. It reads the local
data cache and never downloads (refresh it with ``nflsim fetch``).
"""
