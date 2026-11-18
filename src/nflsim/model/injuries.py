"""Chance each player plays, from his final injury report (Phase 3).

The probabilities live in config ``[injuries]`` and are measured with
``nflsim injury-eval``. Step 2 samples availability from them in every simulation run.
"""

from __future__ import annotations

import polars as pl

from nflsim.config import PRACTICES, REPORT_STATUSES, InjuriesConfig

# Status given to regulars who are not on the injury report at all.
UNLISTED = "unlisted"


def play_probability_table(config: InjuriesConfig) -> pl.DataFrame:
    """Every (regular, status, practice) combination with its ``play_prob``.

    Includes ``status = "unlisted"`` (practice ``none``) for regulars missing from the
    report. Unlisted non-regulars have no row: they are not on the report and did not
    play much last game, so step 2 leaves their projected usage as is.
    """
    rows: list[tuple[bool, str, str, float]] = []
    for regular, table in (
        (True, config.play_probability_regular),
        (False, config.play_probability_other),
    ):
        for status in REPORT_STATUSES:
            for practice in PRACTICES:
                rows.append((regular, status, practice, table[status][practice]))
    rows.append((True, UNLISTED, "none", config.unlisted_regular))
    return pl.DataFrame(
        rows,
        schema={
            "regular": pl.Boolean,
            "status": pl.String,
            "practice": pl.String,
            "play_prob": pl.Float64,
        },
        orient="row",
    )


def attach_play_probability(frame: pl.DataFrame, config: InjuriesConfig) -> pl.DataFrame:
    """Add ``play_prob`` to rows with ``regular``, ``status`` and ``practice`` columns.

    When ``frame`` has a ``position`` column, questionable players at positions in
    ``questionable_position_adjustment`` get that adjustment, clipped to [0, 1].
    """
    out = frame.join(
        play_probability_table(config), on=["regular", "status", "practice"], how="left"
    )
    adjust = config.questionable_position_adjustment
    if adjust and "position" in out.columns:
        delta = pl.col("position").replace_strict(adjust, default=0.0, return_dtype=pl.Float64)
        out = out.with_columns(
            play_prob=pl.when(pl.col("status") == "questionable")
            .then((pl.col("play_prob") + delta).clip(0.0, 1.0))
            .otherwise(pl.col("play_prob"))
        )
    return out
