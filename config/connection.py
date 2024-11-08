import mysql.connector


def get_db_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        database="pa_pens_garbage_website",
        user="root",
        password="",
    )

    return connection


# CREATE TABLE events (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     file_path VARCHAR(255) NOT NULL,
#     detected_object JSON NOT NULL,
#     created_at DATETIME DEFAULT CURRENT_TIMESTAMP
# );
