"""
Live-updating view of the real-time pipeline's predicted vs. realized
reactions, reading from the SQLite table reaction_processor.py writes to.

Usage:
    streamlit run dashboard.py
"""

import os
import sqlite3
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="SEC filing reaction pipeline", layout="wide")
st.title("SEC filing reaction pipeline")
st.caption("Predicted CAR comes from a model that did not beat a naive baseline in "
           "evaluation, so read these numbers as illustrative, not a validated signal.")


@st.fragment(run_every=2)
def show_reactions():
    if not os.path.exists(config.REALTIME_DB_PATH):
        st.info("No data yet. Start replay_producer.py (or the live producers), "
                 "then reaction_processor.py.")
        return

    conn = sqlite3.connect(config.REALTIME_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reactions ORDER BY updated_at DESC", conn)
    conn.close()

    if df.empty:
        st.info("No filings processed yet.")
        return

    df["predicted CAR (%)"] = df["predicted_car"] * 100
    df["realized CAR (%)"] = df["realized_car"] * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Filings tracked", len(df))
    col2.metric("Windows closed", int((df["status"] == "closed").sum()))
    col3.metric("Still accumulating", int((df["status"] != "closed").sum()))

    st.dataframe(
        df[["ticker", "accession_number", "filing_date", "predicted CAR (%)",
            "realized CAR (%)", "days_seen", "status", "updated_at"]],
        width="stretch",
    )

    closed = df[df["status"] == "closed"]
    if len(closed) > 1:
        st.subheader("Predicted vs. realized, closed windows")
        st.scatter_chart(closed, x="predicted CAR (%)", y="realized CAR (%)")


show_reactions()
