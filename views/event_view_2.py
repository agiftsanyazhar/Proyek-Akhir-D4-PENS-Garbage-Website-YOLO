import os
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from controllers.event_controller import index as index_view


def paginate_dataframe(df, page_size=5, page_num=1):
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    return df[start_idx:end_idx]


def app():
    st.title("Application for Detecting Littering Actions using YOLO - Events (2)")

    events = index_view()
    events_df = pd.DataFrame(
        events, columns=["id", "file_path", "detected_object", "created_at"]
    )

    if events_df.empty:
        st.write("No events found")
    else:
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
        events_df["Detected Object"] = events_df["Detected Object"].apply(
            lambda x: list(set(eval(x)))
        )

        page_size = 5
        total_pages = (len(events_df) // page_size) + (
            1 if len(events_df) % page_size != 0 else 0
        )
        current_page = st.session_state.get("current_page", 1)
        events_page = paginate_dataframe(events_df, page_size, current_page)

        for index, row in events_page.iterrows():
            st.write(f"#### Event {row['#']}")

            cols = st.columns([1, 4])
            with cols[0]:
                try:
                    img = Image.open(row["Image"])
                    st.image(img, use_column_width=True)

                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_data = buffered.getvalue()

                    @st.fragment
                    def download_button():
                        st.download_button(
                            label="Download",
                            data=img_data,
                            file_name=os.path.basename(row["Image"]),
                            mime="image/jpeg",
                            key=row["#"],
                        )

                    download_button()

                except FileNotFoundError:
                    st.text("Image not found")

            with cols[1]:
                st.write("**Detected Objects:**", ", ".join(row["Detected Object"]))
                st.write("**Date:**", row["Date"])

            st.write("---")

        st.write(f"Page {current_page} of {total_pages}")
        col1, col2, col3, col4 = st.columns([1, 7, 1, 1])
        with col1:
            if current_page > 1:
                if st.button("First", key="first_bottom"):
                    st.session_state["current_page"] = 1
                    st.rerun()
        with col2:
            if current_page > 1:
                if st.button("Previous", key="previous_bottom"):
                    st.session_state["current_page"] = current_page - 1
                    st.rerun()
        with col3:
            if current_page < total_pages:
                if st.button("Next", key="next_bottom"):
                    st.session_state["current_page"] = current_page + 1
                    st.rerun()
        with col4:
            if current_page < total_pages:
                if st.button("Last", key="last_bottom"):
                    st.session_state["current_page"] = total_pages
                    st.rerun()
