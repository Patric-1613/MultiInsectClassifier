from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load your pre-trained models
model_densenet = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/DenseNet121.keras')
model_inception = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/InceptionV3.keras')
model_resnet = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/ResNet50.keras')
model_mobilenet = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/MobileNetV2.keras')
model_efficientnet = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/EfficientNetB0.keras')
model_customcnn = load_model('/Users/pratikrajmugade/Desktop/API_Dissertation/Moderate_augmentation.keras')  # Custom CNN model

# Define the classes (assuming you have 8 classes)
classes = ['Asian Hornet', 'Asian giant Hornet', 'Carpenter Bee', 'Common wasp', 'European hornet', 'Honey Bee', 'Hoverfly', 'Oriental Hornet']

# Preprocessing function for images
def preprocess_image(img_path, target_size):
    img = Image.open(img_path)
    img = img.resize(target_size)  # Resize image to the target size
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input shape
    return img

# Prediction function
def predict_image(model, img_path, target_size):
    img = preprocess_image(img_path, target_size)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  # Get the class index with highest probability
    return classes[predicted_class[0]]  # Return the predicted class label

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Ensure 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Get the model selected from the dropdown
    selected_model = request.form.get('model')

    # Map the selected model to its corresponding input size
    if selected_model == 'Densenet':
        model = model_densenet
        target_size = (224, 224)
    elif selected_model == 'Inception':
        model = model_inception
        target_size = (299, 299)
    elif selected_model == 'ResNet':
        model = model_resnet
        target_size = (224, 224)
    elif selected_model == 'MobileNet':
        model = model_mobilenet
        target_size = (224, 224)
    elif selected_model == 'EfficientNet':
        model = model_efficientnet
        target_size = (224, 224)
    elif selected_model == 'CustomCNN':
        model = model_customcnn
        target_size = (224, 224)
    else:
        return jsonify({'error': 'Invalid model selected'})

    # Make prediction using the selected model
    predicted_class = predict_image(model, file_path, target_size)

    # Return the predicted class as a JSON response
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
