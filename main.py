# =========================
# Python 3.10.11
# =========================

import streamlit as st
import views.detect_view as dv
import views.event_view as ev

# Streamlit configuration
st.set_page_config(
    page_title="Application for Detecting Littering Actions using YOLO",
    page_icon=":wastebasket:",
)

st.sidebar.title("Dashboard")

page = st.sidebar.selectbox("Select Page", ["Detect", "Events"])

if page is None or page == "Detect":
    dv.app()
elif page == "Events":
    ev.app()
