import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, messagebox, filedialog
import subprocess
import mysql.connector
import os
import pandas as pd  # Add pandas for Excel export
from db_connection import connect_db

# Initialize Main GUI Window
root = ctk.CTk()
root.title("Student Face Recognition System")
root.geometry("1200x700")
ctk.set_appearance_mode("light")  # Set dark mode
ctk.set_default_color_theme("blue")  # Set color theme

# Database Connection
def fetch_students():
    """Retrieve student data including points from MySQL and display in the table."""
    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number, department, course, semester, points FROM students")
    rows = cursor.fetchall()

    student_table.delete(*student_table.get_children())

    for row in rows:
        student_table.insert("", "end", values=row)

    conn.close()

def fetch_attendance():
    """Retrieve attendance data from MySQL and display in the table."""
    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    cursor = conn.cursor()
    cursor.execute("""
        SELECT attendance.id, students.name, students.roll_number, attendance.date, attendance.time
        FROM attendance
        INNER JOIN students ON attendance.student_id = students.id
        ORDER BY attendance.date DESC, attendance.time DESC
    """)
    rows = cursor.fetchall()

    attendance_table.delete(*attendance_table.get_children())

    for row in rows:
        attendance_table.insert("", "end", values=row)

    conn.close()

# Main Frames
main_frame = ctk.CTkFrame(root)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

left_frame = ctk.CTkFrame(main_frame)
left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

right_frame = ctk.CTkFrame(main_frame)
right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=2)
main_frame.grid_rowconfigure(0, weight=1)

# Student Form Input Fields
ctk.CTkLabel(left_frame, text="Student Registration", font=("Arial", 18, "bold")).pack(pady=10)

form_frame = ctk.CTkFrame(left_frame)
form_frame.pack(pady=5, padx=10, fill="x")

labels = ["Student Name:", "Roll Number:", "Department:", "Course:", "Semester:"]
entries = {}

def focus_next_widget(event):
    event.widget.tk_focusNext().focus()
    return "break"

for i, field in enumerate(labels):
    ctk.CTkLabel(form_frame, text=field, font=("Arial", 14)).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = ctk.CTkEntry(form_frame, width=300, height=30, font=("Arial", 14))  # Increased size of entries
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
    entry.bind("<Return>", focus_next_widget)
    entries[field] = entry

# Capture Faces Function
def capture_faces():
    """Capture student images before saving details."""
    roll_no = entries["Roll Number:"].get().strip()
    if not roll_no:
        messagebox.showerror("Input Error", "Enter Roll Number before capturing images!")
        return

    status_label.configure(text="Capturing images...", text_color="blue")
    root.update_idletasks()
    
    result = subprocess.run(["python", "capture_faces.py", roll_no], capture_output=True, text=True)

    if "[Success]" in result.stdout:
        status_label.configure(text="Images Captured Successfully!", text_color="green")
    else:
        status_label.configure(text="Image Capture Failed!", text_color="red")

# Save Student Data
def save_student():
    """Save student details to the database along with captured images."""
    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    cursor = conn.cursor()

    name = entries["Student Name:"].get().strip()
    roll_no = entries["Roll Number:"].get().strip()
    department = entries["Department:"].get().strip()
    course = entries["Course:"].get().strip()
    semester = entries["Semester:"].get().strip()

    if not name or not roll_no:
        messagebox.showwarning("Input Error", "Name and Roll Number are required!")
        return

    # Check if images exist for the student
    dataset_dir = f"dataset/{roll_no}"
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        messagebox.showerror("Capture Error", "No images found! Please capture face images first.")
        return

    try:
        # Insert student details into the database
        cursor.execute("""
            INSERT INTO students (roll_number, name, department, course, semester)
            VALUES (%s, %s, %s, %s, %s)
        """, (roll_no, name, department, course, semester))
        student_id = cursor.lastrowid  # Get inserted student ID

        # Store image paths in student_images table
        for filename in os.listdir(dataset_dir):
            image_path = f"{dataset_dir}/{filename}"
            cursor.execute("INSERT INTO student_images (student_id, image_path) VALUES (%s, %s)", (student_id, image_path))

        conn.commit()
        messagebox.showinfo("Success", "Student data and images saved successfully!")
        fetch_students()  # Refresh student table

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Error: {e}")
    finally:
        conn.close()

# Add Delete Student Function
def delete_student():
    """Delete the selected student's data from all tables and remove their images."""
    selected_item = student_table.selection()
    if not selected_item:
        messagebox.showwarning("Selection Error", "Please select a student to delete.")
        return

    # Get the selected student's roll number
    student_data = student_table.item(selected_item, "values")
    if not student_data:
        messagebox.showerror("Error", "Failed to retrieve student data.")
        return

    roll_no = student_data[2]  # Assuming Roll Number is the 3rd column

    confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete data for Roll Number: {roll_no}?")
    if not confirm:
        return

    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    try:
        cursor = conn.cursor()

        # Delete from student_images table
        cursor.execute("DELETE FROM student_images WHERE student_id = (SELECT id FROM students WHERE roll_number = %s)", (roll_no,))

        # Delete from attendance table
        cursor.execute("DELETE FROM attendance WHERE student_id = (SELECT id FROM students WHERE roll_number = %s)", (roll_no,))

        # Delete from students table
        cursor.execute("DELETE FROM students WHERE roll_number = %s", (roll_no,))

        conn.commit()

        # Remove images from the dataset folder
        dataset_dir = f"dataset/{roll_no}"
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                os.remove(os.path.join(dataset_dir, file))
            os.rmdir(dataset_dir)

        messagebox.showinfo("Success", f"Student data for Roll Number: {roll_no} deleted successfully!")
        fetch_students()  # Refresh the student table
        fetch_attendance()  # Refresh the attendance table
        fetch_top_scores()  # Refresh the top scores table

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", f"Error: {e}")
    finally:
        conn.close()

# Buttons for Student Management (Capture, Save, Refresh, Reset, and Delete Side by Side)
button_frame = ctk.CTkFrame(left_frame)
button_frame.pack(pady=5, padx=10, fill="x")

ctk.CTkButton(button_frame, text="Capture Faces", width=150, height=40, fg_color="#6dbf7d", corner_radius=10, font=("Arial", 14), command=capture_faces).pack(side="left", padx=10, pady=5)
ctk.CTkButton(button_frame, text="Save", width=150, height=40, fg_color="#6dbf7d", corner_radius=10, font=("Arial", 14), command=save_student).pack(side="left", padx=10, pady=5)
ctk.CTkButton(button_frame, text="Refresh", width=150, height=40, fg_color="#6d9ecf", corner_radius=10, font=("Arial", 14), command=lambda: [fetch_students(), fetch_attendance()]).pack(side="left", padx=10, pady=5)
ctk.CTkButton(button_frame, text="Reset", width=150, height=40, fg_color="#f77f7f", corner_radius=10, font=("Arial", 14), command=lambda: reset_form()).pack(side="left", padx=10, pady=5)
ctk.CTkButton(button_frame, text="Delete", width=150, height=40, fg_color="#f77f7f", corner_radius=10, font=("Arial", 14), command=delete_student).pack(side="left", padx=10, pady=5)

# Reset Form Function
def reset_form():
    """Clear all form input fields."""
    for entry in entries.values():
        entry.delete(0, tk.END)
    status_label.configure(text="")

# Status Label
status_label = ctk.CTkLabel(left_frame, text="", font=("Arial", 14))
status_label.pack(pady=5)

# Top 3 Scores
top_scores_frame = ctk.CTkFrame(left_frame)
top_scores_frame.pack(pady=5, padx=10, fill="x")

ctk.CTkLabel(top_scores_frame, text="Top 3 Scores", font=("Arial", 16, "bold")).pack()

top_scores_table = ttk.Treeview(top_scores_frame, columns=("Name", "Roll Number", "Points"), show="headings", height=3)
top_scores_table.heading("Name", text="Name")
top_scores_table.heading("Roll Number", text="Roll Number")
top_scores_table.heading("Points", text="Points")

for col in ("Name", "Roll Number", "Points"):
    top_scores_table.column(col, width=100, anchor="center")

# Increase text size in the table
style = ttk.Style()
style.configure("Treeview.Heading", font=("Arial", 14))
style.configure("Treeview", font=("Arial", 12))

top_scores_table.pack(pady=5, padx=10, fill="x")

def fetch_top_scores():
    """Retrieve top 3 student scores from MySQL and display in the table."""
    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    cursor = conn.cursor()
    cursor.execute("SELECT name, roll_number, points FROM students ORDER BY points DESC LIMIT 3")
    rows = cursor.fetchall()

    top_scores_table.delete(*top_scores_table.get_children())

    for row in rows:
        top_scores_table.insert("", "end", values=row)

    conn.close()

# Student Table
student_frame = ctk.CTkFrame(right_frame)
student_frame.pack(pady=5, padx=10, fill="both", expand=True)

ctk.CTkLabel(student_frame, text="Student Records", font=("Arial", 16, "bold")).pack()

columns = ("ID", "Name", "Roll Number", "Department", "Course", "Semester", "Points")
student_table = ttk.Treeview(student_frame, columns=columns, show="headings", height=10)

for col in columns:
    student_table.heading(col, text=col)
    student_table.column(col, width=120, anchor="center")

# Apply the same style to the student table
student_table.pack(pady=5, padx=10, fill="both", expand=True)

# Attendance Table
attendance_frame = ctk.CTkFrame(right_frame)
attendance_frame.pack(pady=5, padx=10, fill="both", expand=True)

ctk.CTkLabel(attendance_frame, text="Attendance Records", font=("Arial", 16, "bold")).pack()

attendance_columns = ("ID", "Student Name", "Roll Number", "Date", "Time")
attendance_table = ttk.Treeview(attendance_frame, columns=attendance_columns, show="headings", height=10)

for col in attendance_columns:
    attendance_table.heading(col, text=col)
    attendance_table.column(col, width=120, anchor="center")

# Apply the same style to the attendance table
attendance_table.pack(pady=5, padx=10, fill="both", expand=True)

# Export Attendance to Excel
def export_attendance():
    """Export attendance data to an Excel file."""
    conn = connect_db()
    if not conn:
        messagebox.showerror("Database Error", "Failed to connect to the database.")
        return

    cursor = conn.cursor()
    cursor.execute("""
        SELECT attendance.id, students.name, students.roll_number, attendance.date, attendance.time
        FROM attendance
        INNER JOIN students ON attendance.student_id = students.id
        ORDER BY attendance.date DESC, attendance.time DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        messagebox.showinfo("No Data", "No attendance data available to export.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["ID", "Student Name", "Roll Number", "Date", "Time"])

    # Save to Excel
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", ".xlsx"), ("All files", ".*")])
    if file_path:
        df.to_excel(file_path, index=False)
        messagebox.showinfo("Export Successful", f"Attendance data exported successfully to {file_path}")

# Face Recognition Buttons
face_recognition_frame = ctk.CTkFrame(right_frame)
face_recognition_frame.pack(pady=5, padx=10, fill="x")

ctk.CTkButton(face_recognition_frame, text="Train Model", width=150, height=40, fg_color="#6d9ecf", corner_radius=10, font=("Arial", 14), command=lambda: subprocess.run(["python", "train_model.py"])).pack(side="left", padx=10, pady=5)
ctk.CTkButton(face_recognition_frame, text="Start Recognition", width=150, height=40, fg_color="#a891c7", corner_radius=10, font=("Arial", 14), command=lambda: subprocess.run(["python", "recognize_face.py"])).pack(side="left", padx=10, pady=5)
ctk.CTkButton(face_recognition_frame, text="Export Attendance", width=150, height=40, fg_color="#f77f7f", corner_radius=10, font=("Arial", 14), command=export_attendance).pack(side="left", padx=10, pady=5)

# Add a footer with credits
footer_frame = ctk.CTkFrame(root)
footer_frame.pack(fill="x", pady=10)

ctk.CTkLabel(footer_frame, text="Developed by ARG", font=("Arial", 12)).pack()

# Load Data
fetch_students()
fetch_attendance()
fetch_top_scores()

root.mainloop()