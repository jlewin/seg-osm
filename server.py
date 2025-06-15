import os
from flask import Flask, request, jsonify
import json
from flask_cors import CORS

# Import detection functionality from detect.py
from detect import detect_segments
from PIL import Image
import io
from PIL import ImageDraw

app = Flask(__name__)
# Configure static folder
app.static_folder = '.'
app.static_url_path = ''

# Add CORS support for all routes, specifically allowing localhost origins
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:*", 
    "http://127.0.0.1:*",
    "http://0.0.0.0:*"
]}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Redirect to rapid iframe
@app.route('/')
def serve_index():
    return app.redirect('/rapid/parent.html')


# Serve static files from the dist folder at the root
@app.route('/<path:path>')
def serve_dist(path):
    return app.send_static_file(os.path.join('dist', path))


@app.route('/rapid/<path:path>')
def serve_rapid(path):
    # Serve files from the 'rapid' directory
    return app.send_static_file(os.path.join('rapid', path))


@app.route('/detect', methods=['POST'])
def handle_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    props = {}
    if 'props' in request.form:
        try:
            props = json.loads(request.form['props'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON in props field'}), 400
    
    # Open the image from the file stream
    img = Image.open(io.BytesIO(file.read()))
    
    try:
        point = props['point']

        # Process the image
        results = detect_segments(img, point)

        # Draw a circle at the specified point

        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Define circle parameters
        radius = 15
        circle_color = "red"
        # Draw the circle at the specified point
        draw.ellipse(
            [(point[0] - radius, point[1] - radius), 
             (point[0] + radius, point[1] + radius)],
            outline=circle_color,
            width=2
        )

        img.save("dbg/uploaded.png")

        
        # Return the results
        return jsonify({'results': results[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting SAM2 detection server...")
    print("API endpoint available at: http://localhost:5008/detect")
    app.run(host='0.0.0.0', port=5008, debug=True)
