# =========================
# Python 3.10.11
# =========================

import streamlit as st
import detect_copy_1
import event

# Streamlit configuration
st.set_page_config(
    page_title="Application for Detecting Littering Actions using YOLO",
    page_icon=":wastebasket:",
)

# Sidebar Title
st.sidebar.title("Dashboard")

# Sidebar Page Navigation
page = st.sidebar.selectbox("Select Page", ["Detect", "Events"])

# Load the selected page content
if page == "Detect":
    detect_copy_1.app()  # Run the detect page content
elif page == "Events":
    event.app()  # Run the events page content
