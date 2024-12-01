# =========================
# Python 3.10.11
# =========================

import streamlit as st
from views.detect_view import app as dv
from views.event_view_1 import app as ev1
from views.event_view_2 import app as ev2


# Streamlit configuration
st.set_page_config(
    page_title="Real-Time Garbage Detection Application using CNN",
    page_icon=":wastebasket:",
)

st.sidebar.title("Dashboard")
page = st.sidebar.selectbox("Select Page", ["Detect", "Events (1)", "Events (2)"])

if page is None or page == "Detect":
    dv()
elif page == "Events (1)":
    ev1()
elif page == "Events (2)":
    ev2()
