# =========================
# Python 3.10.11
# =========================

import streamlit as st
import views.detect_view as dv
import views.event_view_1 as ev1
import views.event_view_2 as ev2

# Streamlit configuration
st.set_page_config(
    page_title="Application for Detecting Littering Actions using YOLO",
    page_icon=":wastebasket:",
)

st.sidebar.title("Dashboard")
page = st.sidebar.selectbox("Select Page", ["Detect", "Events (1)", "Events (2)"])

if page is None or page == "Detect":
    dv.app()
elif page == "Events (1)":
    ev1.app()
elif page == "Events (2)":
    ev2.app()
