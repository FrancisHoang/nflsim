"""Dashboard entry point: ``nflsim dashboard`` (or ``streamlit run`` on this file)."""

from __future__ import annotations

import streamlit as st

from nflsim.dashboard import game_page, week_page

st.set_page_config(page_title="nflsim", page_icon=":material/sports_football:", layout="wide")

page = st.navigation(
    [
        st.Page(week_page.render, title="Week", icon=":material/calendar_month:", url_path="week"),
        st.Page(game_page.render, title="Game", icon=":material/sports_football:", url_path="game"),
    ]
)
with st.sidebar:
    st.caption(
        "Monte Carlo NFL game simulator. Reads the local data cache; refresh it with "
        "`nflsim fetch`."
    )
page.run()
