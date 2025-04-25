import dlib
import cv2
import os
import sys
import time
import numpy as np

def capture_faces_dlib(roll_number, num_images=150):
    dataset_dir = "dataset"
    student_dir = os.path.join(dataset_dir, roll_number)
    os.makedirs(student_dir, exist_ok=True)

    detector = dlib.get_frontal_face_detector()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[Error] Could not access the webcam!")
        return

    print("[INFO] Capturing 150 images. Move your head in different directions.")
    instructions = [
        "Look straight", "Turn right", "Turn left", "Look up", "Look down",
        "Smile", "Neutral expression"
    ]
    instruction_index = 0
    count = len(os.listdir(student_dir))  # Start from the number of existing images

    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            print("[Error] Failed to capture frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print("[Warning] No face detected.")
            cv2.putText(frame, "Adjust Position!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Capture", frame)
            cv2.waitKey(500)
            continue

        for face in faces:
            x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

            padding = 20
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x2 + padding)
            y_end = min(gray.shape[0], y2 + padding)

            face_img = gray[y_start:y_end, x_start:x_end]
            resized_face = cv2.resize(face_img, (200, 200))

            brightness_factor = np.random.uniform(0.7, 1.3)
            augmented_face = cv2.convertScaleAbs(resized_face, alpha=brightness_factor, beta=0)

            image_path = f"{student_dir}/{roll_number}_{count + 1}.jpg"

            # Check if the image already exists
            if os.path.exists(image_path):
                print(f"[INFO] Image already exists: {image_path}. Skipping...")
                continue

            cv2.imwrite(image_path, augmented_face)

            count += 1
            print(f"[INFO] Image {count} saved: {image_path}")

        cv2.putText(frame, f"Captured: {count}/{num_images}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, instructions[instruction_index], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Face Capture", frame)

        time.sleep(0.3)

        if count % 20 == 0 and instruction_index < len(instructions) - 1:
            instruction_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[Success] All images captured!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[Error] Roll Number not provided!")
    else:
        roll_number = sys.argv[1]
        capture_faces_dlib(roll_number)
