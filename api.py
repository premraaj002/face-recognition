from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Initialize the FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Database setup
db_config = {
    "host": "192.168.1.2",
    "user": "root",
    "password": "highend@009",
    "database": "libary"
}

# Helper Functions
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(face)
    return img_tensor.unsqueeze(0)

def get_embedding(face):
    face_tensor = preprocess_image(face)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding

def compare_embeddings(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy()
    return cosine_similarity(embedding1_np, embedding2)

# Enroll user API
@app.route('/enroll', methods=['POST'])
def enroll_user():
    data = request.json
    name = data.get("name")
    image_data = data.get("image")
    if not name or not image_data:
        return jsonify({"error": "Name and image are required"}), 400

    face_image = Image.open(io.BytesIO(bytearray(image_data)))
    face_embedding = get_embedding(face_image)

    try:
        conn = mysql.connector.connect(**db_config)
        c = conn.cursor()

        # Create table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS users (name VARCHAR(255), embedding BLOB, image BLOB)''')

        # Convert embedding and image to binary
        face_embedding_blob = face_embedding.detach().numpy().tobytes()
        img_bytes = io.BytesIO()
        face_image.save(img_bytes, format='JPEG')
        img_blob = img_bytes.getvalue()

        # Insert into database
        c.execute("INSERT INTO users (name, embedding, image) VALUES (%s, %s, %s)", (name, face_embedding_blob, img_blob))
        conn.commit()
        return jsonify({"message": f"User {name} enrolled successfully!"})
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# Recognize face API
@app.route('/recognize', methods=['POST'])
def recognize_face():
    image_data = request.json.get("image")
    if not image_data:
        return jsonify({"error": "Image is required"}), 400

    face_image = Image.open(io.BytesIO(bytearray(image_data)))
    face_embedding = get_embedding(face_image)

    try:
        conn = mysql.connector.connect(**db_config)
        c = conn.cursor()

        # Retrieve stored embeddings
        c.execute("SELECT name, embedding FROM users")
        users_data = c.fetchall()

        for name, embedding_blob in users_data:
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
            similarity = compare_embeddings(face_embedding, stored_embedding)
            if similarity > 0.7:  # Adjust threshold if needed
                return jsonify({"name": name})

        return jsonify({"name": "Unknown"})
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# Log entry/exit API
@app.route('/log', methods=['POST'])
def log_entry_exit():
    data = request.json
    name = data.get("name")
    entry_exit = data.get("entry_exit")
    if not name or not entry_exit:
        return jsonify({"error": "Name and entry_exit are required"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        c = conn.cursor()

        # Create logs table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS logs (name VARCHAR(255), time DATETIME, entry_exit VARCHAR(10))''')

        # Insert log
        now = datetime.datetime.now()
        c.execute("INSERT INTO logs (name, time, entry_exit) VALUES (%s, %s, %s)", (name, now, entry_exit))
        conn.commit()
        return jsonify({"message": f"Log recorded for {name}"})
    except mysql.connector.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
