import cv2
import torch
import mysql.connector
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import io
import smtplib
from email.mime.text import MIMEText

# Initialize FaceNet model and MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True)

# Function to preprocess the image
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face).unsqueeze(0)

# Function to get face embedding
def get_embedding(face):
    with torch.no_grad():
        return model(preprocess_image(face))

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2):
    return cosine_similarity(embedding1.detach().numpy(), embedding2)

# Database setup
def get_db_connection():
    return mysql.connector.connect(
        host='127.0.0.1',  # Replace with your MySQL host
        user='root',       # Replace with your MySQL username
        password='highend@009',  # Replace with your MySQL password
        database='criminal_db'   # Replace with your database name
    )

# Create tables if they don't exist
def initialize_db():
    with get_db_connection() as conn:
        c = conn.cursor()
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
        conn.commit()

# Function to log criminal detection
def log_detection(criminal_id, location):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO detection_logs (criminal_id, detection_time, location) VALUES (%s, %s, %s)",
                  (criminal_id, datetime.datetime.now(), location))
        conn.commit()

# Function to enroll a new criminal
def enroll_criminal(name, crime_details, face_embedding, face_image):
    face_embedding_blob = face_embedding.detach().numpy().tobytes()
    img_bytes = io.BytesIO()
    face_image.save(img_bytes, format='JPEG')
    img_blob = img_bytes.getvalue()

    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO criminals (name, crime_details, embedding, image) VALUES (%s, %s, %s, %s)",
                  (name, crime_details, face_embedding_blob, img_blob))
        conn.commit()

# Function to retrieve stored criminal data
def get_stored_criminals():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, crime_details, embedding FROM criminals")
        return [
            (id, name, crime_details, np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1))
            for id, name, crime_details, embedding_blob in c.fetchall()
        ]

# Function to send email alert
def send_alert(name, crime_details, location):
    msg = MIMEText(f"Criminal Detected!\nName: {name}\nCrime Details: {crime_details}\nLocation: {location}")
    msg['Subject'] = "Criminal Detection Alert"
    msg['From'] = "your_email@example.com"  # Replace with your email
    msg['To'] = "recipient@example.com"    # Replace with recipient email

    with smtplib.SMTP('smtp.example.com', 587) as server:  # Replace with your SMTP server
        server.login("your_email@example.com", "your_password")  # Replace with your email credentials
        server.sendmail("your_email@example.com", "recipient@example.com", msg.as_string())

# Function to recognize criminals
def recognize_criminal(face_embedding, threshold=0.8):
    for id, name, crime_details, stored_embedding in get_stored_criminals():
        similarity = compare_embeddings(face_embedding, stored_embedding)
        if similarity > threshold:
            return id, name, crime_details, similarity
    return None, "Unknown", "", 0

# Main function for real-time criminal identification
def main():
    initialize_db()
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or RTSP URL for CCTV
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using MTCNN
        faces = mtcnn.detect(frame)
        if faces[0] is not None:
            for face, (x, y, w, h) in zip(faces[0], faces[1]):
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_embedding = get_embedding(face_pil)
                criminal_id, name, crime_details, confidence = recognize_criminal(face_embedding)

                if criminal_id:
                    print(f"Criminal Detected: {name} (Confidence: {confidence:.2f})")
                    log_detection(criminal_id, "Location X")  # Replace with actual location
                    send_alert(name, crime_details, "Location X")  # Send alert

                # Draw rectangle and display results
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Criminal Identification System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()