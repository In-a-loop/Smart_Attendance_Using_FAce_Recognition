import mysql.connector

def connect_db():
    """Establishes a connection with MySQL database."""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",  # Change if necessary
            password="Geetsystem@1234",  # Change this to your MySQL password
            database="students_db",
            charset="utf8mb4"  # Ensures proper encoding
        )
        #print("Checking if ALTER TABLE is being executed...")
        return conn
    except mysql.connector.Error as e:
        print(f"[Error] Database Connection Failed: {e}")
        return None
