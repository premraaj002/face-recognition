import cv2
import torch
import mysql.connector
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
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
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# Function to get face embedding
def get_embedding(face):
    face_tensor = preprocess_image(face)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding

# Function to store user in database
def enroll_user(name, face_image):
    face_embedding = get_embedding(face_image)
    
    # Convert embedding to NumPy array and then to binary format
    face_embedding_np = face_embedding.detach().numpy()
    face_embedding_blob = face_embedding_np.tobytes()

    # Convert image to bytes
    img_bytes = io.BytesIO()
    face_image.save(img_bytes, format='JPEG')
    img_blob = img_bytes.getvalue()

    # Database setup
    conn = mysql.connector.connect(
        host='127.0.0.1',  # Change if necessary
        user='root',  # Replace with your MySQL username
        password='highend@009',  # Replace with your MySQL password
        database='libary'  # Replace with your MySQL database name
    )
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users (name VARCHAR(255) PRIMARY KEY, embedding BLOB, image BLOB)''')
    
    try:
        # Insert user data into the database
        c.execute("INSERT INTO users (name, embedding, image) VALUES (%s, %s, %s)", (name, face_embedding_blob, img_blob))
        conn.commit()
        print(f"User {name} enrolled successfully!")
    except mysql.connector.Error as e:
        print(f"Error: {e}")
    
    conn.close()

# Capture and enroll a user
def capture_and_enroll_user():
    name = input("Enter the user's name: ")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Position the user in front of the camera. Press 'q' to capture the image and enroll the user.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
            
            # Show detected face
            cv2.imshow("Detected Face", face)

            # If 'q' is pressed, capture the image, enroll the user, and break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                enroll_user(name, face_pil)
                cap.release()  # Release the webcam
                cv2.destroyAllWindows()  # Close all OpenCV windows
                return  # Exit the function after enrolling the user

# Run the capture and enrollment process
capture_and_enroll_user()
