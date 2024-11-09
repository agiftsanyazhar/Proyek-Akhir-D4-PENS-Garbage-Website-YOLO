import os
import cv2
import datetime
import config.connection as cn
import json


def index():
    try:
        with cn.get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM events")
                events = cursor.fetchall()
        return events
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
def log_event_to_db(file_path, detected_object):
    try:
        timestamp = datetime.datetime.now()
        connection = cn.get_db_connection()
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
