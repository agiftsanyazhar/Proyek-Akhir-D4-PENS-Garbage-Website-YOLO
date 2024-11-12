import os
import cv2
import datetime
import json
from config.connection import get_db_connection


def index():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM events")
                events = cursor.fetchall()

        # Create a dictionary to hold unique file paths and their associated data
        unique_events = {}
        for event in events:
            file_path = event[1]
            if file_path not in unique_events:
                unique_events[file_path] = event

        # Convert the dictionary values back to a list
        return list(unique_events.values())
    except Exception as e:
        print(f"Error retrieving events from database: {str(e)}")
        return []


# =================================
# Function to save detected events
# =================================
# Function to save detected images
def save_detected_image(frame):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"event_{timestamp}.jpg"
        event_dir = "events"
        os.makedirs(event_dir, exist_ok=True)
        file_path = os.path.join(event_dir, file_name)
        cv2.imwrite(file_path, frame)
        return file_path
    except Exception as e:
        print(f"Error saving detected image: {e}")
        return None


# Function to log event to MySQL database
def store(file_path, detected_object):
    try:
        timestamp = datetime.datetime.now()
        connection = get_db_connection()
        cursor = connection.cursor()
        detected_object_json = json.dumps(detected_object)
        cursor.execute(
            "INSERT INTO events (file_path, detected_object, created_at) VALUES (%s, %s, %s)",
            (file_path, detected_object_json, timestamp),
        )
        connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print(f"Error logging event to DB: {e}")
        return None
    finally:
        cursor.close()
        connection.close()
