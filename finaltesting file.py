import cv2
import torch
import mysql.connector
import numpy as np
from PIL import Image, ImageTk
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import io
import smtplib
from email.mime.text import MIMEText
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

# Initialize FaceNet model and MTCNN
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True)

class CriminalIdentificationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Criminal Identification System")
        self.root.geometry("1200x800")
        
        # Database connection
        self.conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='highend@009',
            database='criminal_db'
        )
        self.initialize_db()
        
        # UI Components
        self.create_widgets()
        self.video_capture = cv2.VideoCapture(0)
        self.is_running = False
        self.current_frame = None
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video feed panel
        self.video_panel = ttk.Label(main_frame)
        self.video_panel.grid(row=0, column=0, padx=10, pady=10, columnspan=2)
        
        # Controls panel
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, padx=10, pady=10)
        
        # Enrollment section
        ttk.Button(controls_frame, text="Enroll New Criminal", command=self.show_enrollment_dialog).pack(pady=5)
        
        # Detection logs
        self.logs_tree = ttk.Treeview(main_frame, columns=("Time", "Name", "Location"), show="headings")
        self.logs_tree.heading("Time", text="Detection Time")
        self.logs_tree.heading("Name", text="Criminal Name")
        self.logs_tree.heading("Location", text="Location")
        self.logs_tree.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        # Start/Stop button
        self.toggle_btn = ttk.Button(controls_frame, text="Start Detection", command=self.toggle_detection)
        self.toggle_btn.pack(pady=5)
        
    def initialize_db(self):
        with self.conn.cursor() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS criminals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                crime_details TEXT,
                embedding BLOB,
                image BLOB
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS detection_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                criminal_id INT,
                detection_time DATETIME,
                location VARCHAR(255)
            )''')
            self.conn.commit()

    def show_enrollment_dialog(self):
        enroll_window = tk.Toplevel(self.root)
        enroll_window.title("Enroll New Criminal")
        
        ttk.Label(enroll_window, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        name_entry = ttk.Entry(enroll_window)
        name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(enroll_window, text="Crime Details:").grid(row=1, column=0, padx=5, pady=5)
        crime_entry = ttk.Entry(enroll_window)
        crime_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(enroll_window, text="Upload/Capture Image:").grid(row=2, column=0, padx=5, pady=5)
        self.enroll_image = None
        
        def capture_image():
            ret, frame = self.video_capture.read()
            if ret:
                self.enroll_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                messagebox.showinfo("Success", "Image captured successfully!")
        
        def upload_image():
            file_path = filedialog.askopenfilename()
            if file_path:
                self.enroll_image = Image.open(file_path)
                messagebox.showinfo("Success", "Image uploaded successfully!")
                
        ttk.Button(enroll_window, text="Capture from Camera", command=capture_image).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(enroll_window, text="Upload Image", command=upload_image).grid(row=3, column=1, padx=5, pady=5)
        
        def save_enrollment():
            if self.enroll_image:
                try:
                    face_embedding = get_embedding(self.enroll_image)
                    enroll_criminal(
                        name_entry.get(),
                        crime_entry.get(),
                        face_embedding,
                        self.enroll_image
                    )
                    messagebox.showinfo("Success", "Criminal enrolled successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Enrollment failed: {str(e)}")
            else:
                messagebox.showwarning("Warning", "Please capture/upload an image first!")
                
        ttk.Button(enroll_window, text="Save", command=save_enrollment).grid(row=4, column=1, padx=5, pady=10)

    def toggle_detection(self):
        if not self.is_running:
            self.is_running = True
            self.toggle_btn.config(text="Stop Detection")
            threading.Thread(target=self.detection_loop, daemon=True).start()
        else:
            self.is_running = False
            self.toggle_btn.config(text="Start Detection")

    def detection_loop(self):
        while self.is_running:
            ret, frame = self.video_capture.read()
            if ret:
                faces = mtcnn.detect(frame)
                if faces[0] is not None:
                    for face, (x, y, w, h) in zip(faces[0], faces[1]):
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_embedding = get_embedding(face_pil)
                        criminal_id, name, crime_details, confidence = recognize_criminal(face_embedding)
                        
                        if criminal_id:
                            self.log_detection(criminal_id, "Location X")
                            self.send_alert(name, crime_details, "Location X")
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_video_panel()
                
    def update_video_panel(self):
        if self.current_frame is not None:
            img = Image.fromarray(self.current_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.configure(image=imgtk)

    def log_detection(self, criminal_id, location):
        with self.conn.cursor() as c:
            c.execute("INSERT INTO detection_logs (criminal_id, detection_time, location) VALUES (%s, %s, %s)",
                     (criminal_id, datetime.datetime.now(), location))
            self.conn.commit()
        self.update_logs()

    def send_alert(self, name, crime_details, location):
        # Implement your alert logic here
        print(f"Alert: {name} detected at {location}")

    def update_logs(self):
        for row in self.logs_tree.get_children():
            self.logs_tree.delete(row)
            
        with self.conn.cursor() as c:
            c.execute("SELECT detection_time, name, location FROM detection_logs JOIN criminals ON detection_logs.criminal_id = criminals.id")
            for log in c.fetchall():
                self.logs_tree.insert("", "end", values=log)

# Helper functions
def get_embedding(face):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_tensor = transform(face).unsqueeze(0)
    with torch.no_grad():
        return model(face_tensor)

def compare_embeddings(embedding1, embedding2):
    return cosine_similarity(embedding1.detach().numpy(), embedding2)

def enroll_criminal(name, crime_details, face_embedding, face_image):
    face_embedding_blob = face_embedding.detach().numpy().tobytes()
    img_bytes = io.BytesIO()
    face_image.save(img_bytes, format='JPEG')
    img_blob = img_bytes.getvalue()
    
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='highend@009',
        database='criminal_db'
    )
    
    with conn.cursor() as c:
        c.execute("INSERT INTO criminals (name, crime_details, embedding, image) VALUES (%s, %s, %s, %s)",
                 (name, crime_details, face_embedding_blob, img_blob))
        conn.commit()
    conn.close()

def recognize_criminal(face_embedding, threshold=0.8):
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='highend@009',
        database='criminal_db'
    )
    
    with conn.cursor() as c:
        c.execute("SELECT id, name, crime_details, embedding FROM criminals")
        criminals = [
            (id, name, crime_details, np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1))
            for id, name, crime_details, embedding_blob in c.fetchall()
        ]
    
    for id, name, crime_details, stored_embedding in criminals:
        similarity = compare_embeddings(face_embedding, stored_embedding)
        if similarity > threshold:
            conn.close()
            return id, name, crime_details, similarity[0][0]
    
    conn.close()
    return None, "Unknown", "", 0

if __name__ == "__main__":
    root = tk.Tk()
    app = CriminalIdentificationSystem(root)
    root.mainloop()