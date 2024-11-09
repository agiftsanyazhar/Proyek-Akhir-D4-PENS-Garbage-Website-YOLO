import streamlit as st
import pandas as pd
from controllers.event_controller import index


def app():
    st.title("Application for Detecting Littering Actions using YOLO - Events (1)")

    events = index()
    events_df = pd.DataFrame(
        events, columns=["id", "file_path", "detected_object", "created_at"]
    )

    events_df.rename(
        columns={
            "id": "#",
            "file_path": "Image",
            "detected_object": "Detected Object",
            "created_at": "Date",
        },
        inplace=True,
    )
    events_df["#"] = range(1, len(events_df) + 1)
    events_df["Image"] = events_df["Image"].apply(lambda x: x.split("\\")[-1])
    events_df["Detected Object"] = events_df["Detected Object"].apply(
        lambda x: list(set(eval(x)))
    )

    st.dataframe(
        events_df[["#", "Image", "Detected Object", "Date"]],
        column_config={
            "#": "#",
            "Image": "Image",
            "Detected Object": "Detected Objects",
            "Date": "Date",
        },
        hide_index=True,
    )
