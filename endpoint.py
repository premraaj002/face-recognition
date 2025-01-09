from flask import Flask, request, jsonify
import reco  # Import your face recognition code here

app = Flask(__name__)

@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    # Access the uploaded image
    image_file = request.files.get('image')
    if image_file:
        # Process the image with face recognition
        # Convert to the format expected by your recognition code if necessary
        recognition_result = reco.recognize(image_file)
        return jsonify({"status": "success", "result": recognition_result})
    else:
        return jsonify({"status": "error", "message": "No image provided"}), 400
@app.route('/')
def home():
    return "Welcome to the Face Recognition API"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Open for local testing
