"""Streamlit dashboard over the saved results: ``streamlit run greenqueue/demo/app.py``.

Set GQ_RESULTS to point at another results folder. Every number comes from a CSV written by an
experiment; nothing is computed here. The scheduling tables use a synthetic job-energy estimator.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly.express as px
import streamlit as st

from greenqueue.demo.tables import alpha_vs_violation, load_tables

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7a7a7a"]

st.set_page_config(page_title="GreenQueue results", layout="wide")
st.title("GreenQueue: uncertainty-aware carbon scheduling (GB)")
st.caption(
    "Results of the saved experiments. Scheduling energy/duration estimates are synthetic; "
    "GPU power is assumed. See the README for limitations."
)
tables = load_tables(Path(os.environ.get("GQ_RESULTS", "results")))

tab_f, tab_s, tab_a, tab_m, tab_e = st.tabs(
    ["Forecast coverage", "Scheduling", "Alpha tracking", "Average vs marginal", "Energy models"]
)


def missing(name: str) -> None:
    st.info(f"No {name} results found. Run the experiment first (`greenqueue run <exp>`).")


with tab_f:
    df = tables["forecast"]
    if df is None:
        missing("E2")
    else:
        st.write("Coverage of nominal 90% intervals over the test period (target 0.90).")
        fig = px.bar(df, x="method", y="coverage", color="method", color_discrete_sequence=COLORS)
        fig.add_hline(y=0.9, line_dash="dash")
        fig.update_layout(showlegend=False, yaxis_range=[0.6, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, width="stretch")

with tab_s:
    df = tables["scheduling"]
    if df is None:
        missing("E4")
    else:
        metric = st.radio(
            "Metric", ["saving_vs_s0", "violation_rate", "mean_wait_h"], horizontal=True
        )
        fig = px.bar(df, x="scheduler", y=metric, color="scheduler", color_discrete_sequence=COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, width="stretch")

with tab_a:
    df = tables["alpha"]
    if df is None:
        missing("E5")
    else:
        a = alpha_vs_violation(df)
        fig = px.line(
            a, x="alpha", y="violation_rate", color="scenario", markers=True,
            color_discrete_sequence=COLORS,
        )  # fmt: skip
        fig.add_shape(type="line", x0=0, y0=0, x1=0.2, y1=0.2, line={"dash": "dash"})
        st.write("Robust MPC deadline-violation rate against the target alpha (dashed = y = x).")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(a, width="stretch")

with tab_m:
    df = tables["accounting"]
    if df is None:
        missing("E8")
    else:
        st.write(
            "Saving vs ASAP when the same schedules are re-scored with marginal-emission "
            "PROXIES (not measured marginal data)."
        )
        fig = px.bar(
            df[df["scheduler"] != "S0_asap"], x="scheduler", y="saving_vs_s0",
            color="accounting", barmode="group", color_discrete_sequence=COLORS,
        )  # fmt: skip
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, width="stretch")

with tab_e:
    df = tables["energy"]
    if df is None:
        missing("E3")
    else:
        st.warning("One Tesla T4, 23 configurations: errors are large and intervals are infinite.")
        st.dataframe(df, width="stretch")
