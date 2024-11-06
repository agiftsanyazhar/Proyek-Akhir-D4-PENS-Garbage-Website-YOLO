import streamlit as st
import detect
import event

# Streamlit configuration
st.set_page_config(
    page_title="Application for Detecting Littering Actions using YOLO",
    page_icon=":wastebasket:",
)

# Sidebar Title
st.sidebar.title("Dashboard")

# Sidebar Page Navigation
page = st.sidebar.selectbox("Select Page", ["Detect"])

# Load the selected page content
if page == "Detect":
    detect.app()  # Run the detect page content
elif page == "Events":
    event.app()  # Run the events page content
