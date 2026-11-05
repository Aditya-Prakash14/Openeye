"""OpenEye Sentinel dashboard. Reads out/events.jsonl from pipeline.py.

Run:  streamlit run sentinel/dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

EVENTS = Path(__file__).resolve().parent.parent / "out" / "events.jsonl"

st.title("OpenEye Sentinel")

if not EVENTS.exists():
    st.warning("No events yet. Run: python sentinel/pipeline.py --source synthetic --model motion")
    st.stop()

events = [json.loads(line) for line in EVENTS.read_text().splitlines() if line.strip()]
df = pd.DataFrame(events)

c1, c2, c3 = st.columns(3)
c1.metric("events", len(df))
c1.metric("zone entries", int((df["type"] == "zone_enter").sum()))
exits = df[df["type"] == "zone_exit"]
if len(exits):
    c2.metric("avg dwell (s)", round(exits["dwell_s"].mean(), 1))
c2.metric("speed events", int((df["type"] == "speed").sum()))

left, right = st.columns(2)
with left:
    st.subheader("entries per zone")
    st.bar_chart(df[df["type"] == "zone_enter"]["zone"].value_counts())
with right:
    st.subheader("event types")
    st.bar_chart(df["type"].value_counts())

st.subheader("timeline")
if "zone" in df:
    st.scatter_chart(df[df["type"] != "speed"], x="t", y="zone")

st.subheader("raw events (latest 200)")
st.dataframe(df.tail(200))
