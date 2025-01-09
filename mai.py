import cv2
import torch
import mysql.connector
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import io

# Initialize the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to preprocess the image (resize, convert to tensor, normalize)
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(face)
    img_tensor = img_tensor.unsqueeze(0)  
    return img_tensor

# Function to get face embedding
def get_embedding(face):
    face_tensor = preprocess_image(face)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding

# Function to compare embeddings and return cosine similarity
def compare_embeddings(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy()  # Convert PyTorch tensor to NumPy
    return cosine_similarity(embedding1_np, embedding2)

# Database setup
conn = mysql.connector.connect(
    host='127.0.0.1',  # Change if necessary
    user='root',  # Replace with your MySQL username
    password='highend@009',  # Replace with your MySQL password
    database='libary'  # Replace with your MySQL database name
)
c = conn.cursor()

# Create tables for logs and users if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS logs (name VARCHAR(255), time DATETIME, entry_exit VARCHAR(10))''')
c.execute('''CREATE TABLE IF NOT EXISTS users (name VARCHAR(255) , embedding BLOB, image BLOB)''')
conn.commit()

# Function to log entry/exit
def log_entry_exit(name, entry_exit):
    now = datetime.datetime.now()
    c.execute("INSERT INTO logs (name, time, entry_exit) VALUES (%s, %s, %s)", (name, now, entry_exit))
    conn.commit()

# Function to enroll a new user by capturing their face embedding and image
def enroll_user(name, face_embedding, face_image):
    face_embedding_np = face_embedding.detach().numpy()  # Convert to NumPy array
    face_embedding_blob = face_embedding_np.tobytes()  # Convert to binary format

    # Convert image to bytes
    img_bytes = io.BytesIO()
    face_image.save(img_bytes, format='JPEG')
    img_blob = img_bytes.getvalue()

    c.execute("INSERT INTO users (name, embedding, image) VALUES (%s, %s, %s)", (name, face_embedding_blob, img_blob))
    conn.commit()

# Function to retrieve stored embeddings and images from the database
def get_stored_data():
    c.execute("SELECT * FROM users")
    users_data = c.fetchall()

    stored_data = []
    for name, embedding_blob, image_blob in users_data:
        embedding_np = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)  # Convert embedding
        image = Image.open(io.BytesIO(image_blob))  # Convert to PIL Image
        stored_data.append((name, embedding_np, image))
    
    return stored_data

# Retrieve the stored data from the database
stored_data = get_stored_data()

# Function to recognize faces using stored embeddings and images from the database
def recognize_face(face_embedding):
    for name, stored_embedding, stored_image in stored_data:
        similarity = compare_embeddings(face_embedding, stored_embedding)
        print(f"Similarity with {name}: {similarity}")  # Debugging output

        if similarity > 0.7:  # Adjust threshold if needed
            return name  # Only return the name, no image popping
    
    return "Unknown"

# Initialize webcam for real-time face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(f"Detected {len(faces)} face(s)")  # Debugging output
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Show detected face
        cv2.imshow("Detected Face", face)

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
        face_embedding = get_embedding(face_pil)
        print(f"Face embedding: {face_embedding}")  # Debugging output
        
        name = recognize_face(face_embedding)  # Only return name
        if name == "Unknown":
            print("Unknown face detected.")
        else:
            print(f"Recognized {name}. Logging entry/exit.")
            log_entry_exit(name, "Entry")

        # Draw rectangle around face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Library Entry System', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
