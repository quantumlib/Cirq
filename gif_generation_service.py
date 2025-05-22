import os
import uuid
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw # Pillow for image manipulation
import imageio # For GIF generation

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads/' # Where original images are stored
GENERATED_GIFS_FOLDER = 'generated_gifs/'
GIF_DURATION_SECONDS = 5
GIF_FPS = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_GIFS_FOLDER'] = GENERATED_GIFS_FOLDER

# Create output folder if it doesn't exist
if not os.path.exists(GENERATED_GIFS_FOLDER):
    os.makedirs(GENERATED_GIFS_FOLDER)
if not os.path.exists(UPLOAD_FOLDER): # Should be created by upload service, but good to ensure
    os.makedirs(UPLOAD_FOLDER)

def placeholder_quantum_effect(image_path, num_frames):
    """
    Placeholder for the quantum-resistant algorithm.
    Generates a sequence of frames from an image.
    For now, it creates a simple animation (e.g., rotating or fading).
    """
    frames = []
    try:
        img = Image.open(image_path).convert("RGBA") # Ensure RGBA for transparency if needed
        width, height = img.size

        for i in range(num_frames):
            frame = img.copy()
            draw = ImageDraw.Draw(frame)
            
            # Example: Simple rotation effect
            angle = (i / float(num_frames)) * 30 # Rotate up to 30 degrees
            rotated_frame = img.rotate(angle, expand=False, fillcolor=(255,255,255,0)) # Fill with transparent white

            # Example: Add a changing number overlay
            # draw_rotated = ImageDraw.Draw(rotated_frame)
            # text = f"Frame {i+1}"
            # text_width, text_height = draw_rotated.textsize(text) # textsize is deprecated, use textbbox in newer Pillow
            # text_x = (width - text_width) / 2
            # text_y = (height - text_height) / 2
            # try: # textbbox is preferred in newer Pillow
            #     bbox = draw_rotated.textbbox((text_x, text_y), text)
            #     draw_rotated.text((text_x, text_y), text, fill="black")
            # except AttributeError: # Fallback for older Pillow
            #      draw_rotated.text((text_x, text_y), text, fill="black")


            # Simpler effect for placeholder: fade in/out effect by changing alpha
            # This requires image to be RGBA
            alpha_img = Image.new("L", img.size, 0) # Create a new alpha channel
            # Sinusoidal fade: 0 -> max_alpha -> 0
            # max_alpha = 150 # Semi-transparent
            # current_alpha = int(max_alpha * (1 + math.sin(math.pi * i / num_frames - math.pi/2)) / 2)
            # alpha_img = Image.new("L", img.size, current_alpha)
            # frame.putalpha(alpha_img)

            # Simplest: use the rotated frame
            frames.append(rotated_frame)
            
        return frames
    except Exception as e:
        print(f"Error in placeholder_quantum_effect: {e}")
        return []


@app.route('/generate_gif', methods=['POST'])
def generate_gif_route():
    data = request.get_json()
    if not data or 'image_id' not in data:
        return jsonify({"success": False, "error": "Missing image_id in request"}), 400

    image_id = data['image_id']
    
    # Security: Basic sanitization. Ensure image_id is just a filename.
    # A more robust approach might involve checking against a database of uploaded files.
    if not image_id or '/' in image_id or '..' in image_id:
        return jsonify({"success": False, "error": "Invalid image_id format"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)

    if not os.path.exists(image_path):
        return jsonify({"success": False, "error": f"Image not found: {image_id}"}), 404

    num_frames = GIF_DURATION_SECONDS * GIF_FPS
    
    # --- Image Processing Core & Quantum Placeholder ---
    # In a real scenario, this could be a complex operation.
    # For computationally intensive tasks, consider background workers (e.g., Celery).
    try:
        frames = placeholder_quantum_effect(image_path, num_frames)
        if not frames:
            return jsonify({"success": False, "error": "Failed to generate frames from image"}), 500
    except Exception as e:
        # Log exception e
        return jsonify({"success": False, "error": f"Error during image processing: {str(e)}"}), 500

    # --- GIF Generation (imageio) ---
    gif_filename = f"{uuid.uuid4()}.gif"
    gif_filepath = os.path.join(app.config['GENERATED_GIFS_FOLDER'], gif_filename)

    try:
        # Ensure frames are in a format imageio understands (e.g., numpy arrays or PIL Images)
        # Pillow images are fine.
        imageio.mimsave(gif_filepath, frames, fps=GIF_FPS, subrectangles=True) # subrectangles for optimization
    except Exception as e:
        # Log exception e
        return jsonify({"success": False, "error": f"Failed to generate GIF: {str(e)}"}), 500

    return jsonify({
        "success": True, 
        "gif_id": gif_filename,
        "message": "GIF generated successfully."
    }), 201

if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like Gunicorn or uWSGI
    # and consider background workers for the GIF generation part.
    app.run(debug=True, port=5001)
