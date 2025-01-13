from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from flask_cors import CORS
import numpy as np
from nbformat import read
from nbclient import NotebookClient
from io import BytesIO
import model

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

def load_function_from_notebook(notebook_path, function_name):
    # Read the notebook
    with open(notebook_path) as f:
        nb = read(f, as_version=4)
    
    # Execute the notebook
    client = NotebookClient(nb)
    client.execute()
    
    # Get the function from the notebook's global namespace
    return client.nb.cells[-1].outputs[0].data[function_name]


# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Get the uploaded file
    file = request.files['file']

    # Open the image using PIL without saving to disk
    image = Image.open(BytesIO(file.read()))

    

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Resize and preprocess the image (ensure this matches your training pipeline)
        image = image.resize((64, 64)) 
        image = np.array(image)
        output_image = model.predict(image)

        return jsonify({'prediction': output_image}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/index', methods=['GET'])
def webPage():
    return render_template("index.html")



# Run the app
if __name__ == '__main__':
    app.run(debug=True,port=5000)
