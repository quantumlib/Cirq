import os
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image part in the request"}), 400
    
    file = request.files['image']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File type not allowed"}), 400

    # Validate file size (Flask automatically handles MAX_CONTENT_LENGTH for request size)
    # For more granular control, one might need to stream and check, but Flask's default is often sufficient.

    if file:
        # Sanitize filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(filepath)
            return jsonify({"success": True, "file_id": unique_filename}), 201
        except Exception as e:
            # Log the exception e for debugging
            return jsonify({"success": False, "error": "Failed to save file"}), 500

    return jsonify({"success": False, "error": "Unknown error"}), 500

if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(debug=True, port=5000)
