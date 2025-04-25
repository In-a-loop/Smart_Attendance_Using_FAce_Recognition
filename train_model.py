'''import cv2
import numpy as np
import os
import random
from db_connection import connect_db

def train_face_recognizer():
    """Trains the LBPH Face Recognition Model and calculates accuracy."""
    
    dataset_dir = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    faces = []
    labels = []

    conn = connect_db()
    cursor = conn.cursor()

    for student_folder in os.listdir(dataset_dir):
        student_path = os.path.join(dataset_dir, student_folder)
        
        if not os.path.isdir(student_path):
            continue  

        cursor.execute("SELECT id FROM students WHERE roll_number = %s", (student_folder,))
        student_id = cursor.fetchone()

        if not student_id:
            print(f"[Warning] No database entry for {student_folder}. Skipping...")
            continue

        student_id = student_id[0]

        student_faces = []
        for image_name in os.listdir(student_path):
            image_path = os.path.join(student_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"[Error] Failed to read {image_path}. Skipping...")
                continue

            # Apply histogram equalization for better contrast
            img = cv2.equalizeHist(img)
            student_faces.append((img, student_id))

        # Shuffle dataset for randomness
        random.shuffle(student_faces)

        # Split data: 80% for training, 20% for testing
        split_index = int(0.8 * len(student_faces))
        train_data = student_faces[:split_index]
        test_data = student_faces[split_index:]

        for img, label in train_data:
            faces.append(img)
            labels.append(label)

    conn.close()

    if len(faces) == 0 or len(labels) == 0:
        print("[Error] No images found for training!")
        return False

    labels = np.array(labels)

    print("[INFO] Training LBPH model...")
    recognizer.train(faces, labels)

    recognizer.save("trained_model.yml")
    print("[Success] Model trained and saved as 'trained_model.yml'!")

    # Evaluate accuracy on test set
    correct_predictions = 0
    total_tests = 0

    for img, actual_label in test_data:
        predicted_label, confidence = recognizer.predict(img)
        total_tests += 1
        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    print(f"[INFO] Model Accuracy: {accuracy:.2f}%")

    return accuracy

if __name__ == "__main__":
    accuracy = train_face_recognizer()
    print(f"Final Model Accuracy: {accuracy:.2f}%")


# save_embeddings_dlib.py

import os
import dlib
import cv2
import numpy as np
import pickle
from db_connection import connect_db

# Load models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None

    face = faces[0]
    shape = shape_predictor(image, face)
    face_embedding = np.array(face_encoder.compute_face_descriptor(image, shape))
    return face_embedding

# Updated portion of save_embeddings_dlib.py (aka train_model.py)

def generate_embeddings(dataset_dir="dataset"):
    embeddings = []
    labels = []

    for student_roll in os.listdir(dataset_dir):
        student_dir = os.path.join(dataset_dir, student_roll)
        if not os.path.isdir(student_dir):
            continue

        # 👇 Fetch student name from database
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students WHERE roll_number = %s", (student_roll,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            print(f"[Warning] No name found for roll: {student_roll}")
            continue

        student_name = result[0]
        label = f"{student_roll} - {student_name}"  # Combine roll + name

        for img_file in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            embedding = get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label)
                #print(f"[INFO] Processed: {img_path} as {label}")

    # Save embeddings with combined labels
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((embeddings, labels), f)
    print("[Success] Embeddings saved to embeddings.pkl")


    # Save to disk
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((embeddings, labels), f)
    print("[Success] Embeddings saved to embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings()
'''
import os
import dlib
import cv2
import numpy as np
import pickle
import random
from db_connection import connect_db

# ========== Load Models ==========
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ========== Helper Function ==========
def get_face_embedding(image):
    """Extracts face embeddings using dlib."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None

    face = faces[0]
    shape = shape_predictor(image, face)
    face_embedding = np.array(face_encoder.compute_face_descriptor(image, shape))
    return face_embedding
print("Model training started.")
# ========== Generate Embeddings ==========
def generate_embeddings(dataset_dir="dataset"):
    """Generates embeddings for all images in the dataset."""
    embeddings = []
    labels = []

    # Load existing embeddings if available
    processed_images = set()
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            existing_embeddings, existing_labels = pickle.load(f)
            embeddings.extend(existing_embeddings)
            labels.extend(existing_labels)
            # Extract already processed image paths
            processed_images = {label.split(" - ")[0] for label in existing_labels}
        print(f"[INFO] Loaded {len(processed_images)} already processed images.")

    for student_roll in os.listdir(dataset_dir):
        student_dir = os.path.join(dataset_dir, student_roll)
        if not os.path.isdir(student_dir):
            continue

        # Fetch student name from database
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students WHERE roll_number = %s", (student_roll,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            print(f"[Warning] No name found for roll: {student_roll}")
            continue

        student_name = result[0]
        label = f"{student_roll} - {student_name}"  # Combine roll + name

        for img_file in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_file)

            # Skip already processed images
            if img_path in processed_images:
                print(f"[INFO] Skipping already processed image: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Failed to read image: {img_path}")
                continue

            embedding = get_face_embedding(img)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label)
                processed_images.add(img_path)  # Mark image as processed
               # print(f"[INFO] Processed: {img_path} as {label}")
    
    # Save updated embeddings with combined labels
    with open("embeddings.pkl", "wb") as f:
        pickle.dump((embeddings, labels), f)
    print("[Success] Embeddings saved to embeddings.pkl")



# ========== Main ==========
if __name__ == "__main__":
    print("[INFO] Generating embeddings...")
    generate_embeddings()

    