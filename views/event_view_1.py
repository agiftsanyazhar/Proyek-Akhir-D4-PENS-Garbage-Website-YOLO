import streamlit as st
import controllers.event_controller as ec
import pandas as pd


def app():
    st.title("Application for Detecting Littering Actions using YOLO - Events")
    events = ec.index()
    events_df = pd.DataFrame(
        events, columns=["id", "file_path", "detected_object", "created_at"]
    )
    events_df["id"] = range(1, len(events_df) + 1)
    events_df.rename(
        columns={
            "id": "#",
            "file_path": "Image",
            "detected_object": "Detected Object",
            "created_at": "Timestamp",
        },
        inplace=True,
    )
    events_df["Detected Object"] = events_df["Detected Object"].apply(
        lambda x: list(set(eval(x)))
    )
    st.dataframe(
        events_df[["#", "Image", "Detected Object", "Timestamp"]],
        column_config={
            "#": "#",
            "Image": "Image",
            "Detected Object": "Detected Objects",
            "Timestamp": "Timestamp",
        },
        hide_index=True,
    )
