import streamlit as st
import controllers.event_controller as ec


def app():
    st.title("Application for Detecting Littering Actions using YOLO - Events")

    events = ec.index()

    st.dataframe(events)
