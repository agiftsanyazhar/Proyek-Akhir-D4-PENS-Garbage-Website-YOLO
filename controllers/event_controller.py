import os
import cv2
import datetime
import config.connection as cn
import json


def index():
    connection = cn.get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM events")
    events = cursor.fetchall()
    cursor.close()
    connection.close()

    return events


# =================================
# Function to save detected events
# =================================
# Function to save detected images
def save_detected_image(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"event_{timestamp}.jpg"
    event_dir = "events"
    os.makedirs(event_dir, exist_ok=True)
    file_path = os.path.join(event_dir, file_name)
    cv2.imwrite(file_path, frame)

    return file_path


# Function to log event to MySQL database
def log_event_to_db(file_path, detected_object):
    timestamp = datetime.datetime.now()
    connection = cn.get_db_connection()
    cursor = connection.cursor()
    detected_object_json = json.dumps(detected_object)
    cursor.execute(
        "INSERT INTO events (file_path, detected_object, created_at) VALUES (%s, %s, %s)",
        (file_path, detected_object_json, timestamp),
    )
    connection.commit()
    cursor.close()
    connection.close()
