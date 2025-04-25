import dlib
import cv2
import numpy as np
import pickle
from scipy.spatial import distance
from datetime import datetime
from db_connection import connect_db

# ========== Load Dlib Models ==========
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ========== Load Known Embeddings ==========
with open("embeddings.pkl", "rb") as f:
    known_embeddings, known_labels = pickle.load(f)

# ========== Connect to DB ==========
conn = connect_db()
if not conn:
    print("[ERROR] Database connection failed.")
    exit()
cursor = conn.cursor()

# ========== Record Attendance Function ==========
def record_attendance(student_id):
    """Marks attendance and assigns points based on check-in time."""
    try:
        cursor = conn.cursor()
        today = datetime.today().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')

        print(f"[DEBUG] Attempting to mark attendance for Student ID: {student_id} at {current_time} on {today}")

        # Check if attendance is already marked
        cursor.execute("SELECT * FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
        already_marked = cursor.fetchone()

        print(f"[DEBUG] Attendance Query Result: {already_marked}")

        if not already_marked:
            cursor.execute("INSERT INTO attendance (student_id, date, time) VALUES (%s, %s, %s)", (student_id, today, current_time))
            conn.commit()
            print("[INFO] Attendance Recorded Successfully!")

            # Award points based on check-in time
            check_in_time = datetime.strptime(current_time, "%H:%M:%S").time()
            points = 0
            if check_in_time < datetime.strptime("09:00:00", "%H:%M:%S").time():
                points += 10  # On-time arrival
            elif check_in_time < datetime.strptime("10:00:00", "%H:%M:%S").time():
                points += 5   # Late arrival

            # Update student points in the database
            cursor.execute("UPDATE students SET points = points + %s WHERE id = %s", (points, student_id))
            conn.commit()
            print(f"[INFO] Attendance marked for Student ID {student_id} | Points Awarded: {points}")

    except Exception as e:
        print(f"[ERROR] Failed to record attendance: {e}")

# ========== Recognition Function ==========
def recognize_face(frame):
    faces = face_detector(frame)
    for face in faces:
        x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        shape = shape_predictor(frame, face)
        embedding = np.array(face_encoder.compute_face_descriptor(frame, shape))

        distances = [distance.euclidean(embedding, e) for e in known_embeddings]
        min_distance = min(distances)
        label = "Unknown"
        color = (0, 0, 255)

        if min_distance < 0.6:
            label = known_labels[distances.index(min_distance)]
            color = (0, 255, 0)

            # Debug the extracted label
            print(f"[DEBUG] Raw Label from Recognition: {label}")

            try:
                # Extract roll number from label and remove extra spaces
                roll_no = label.split("-")[0].strip()
                print(f"[DEBUG] Extracted Roll Number: '{roll_no}'")

                # Query using roll number instead of name
                cursor.execute("SELECT id FROM students WHERE roll_number = %s", (roll_no,))
                student_result = cursor.fetchone()

                print(f"[DEBUG] Retrieved Student ID for Roll Number '{roll_no}': {student_result}")

                if student_result:
                    student_id = student_result[0]
                    record_attendance(student_id)
                else:
                    print(f"[ERROR] No student found for Roll Number '{roll_no}'")

            except Exception as e:
                print(f"[ERROR] Label processing failed: {e}")

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({min_distance:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ========== Start Webcam ==========
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = recognize_face(frame)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
