import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model("skin_cancer_detection_model.h5")

# Define a function to preprocess the input image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions
def predict_image(model, img_path):
    preprocessed_image = preprocess_image(img_path)
    prediction = model.predict(preprocessed_image)
    return prediction

# Path to the input image
img_path = "ISIC_0015251_downsampled_640x426.jpg"

# Make prediction
prediction = predict_image(model, img_path)

# Decode the prediction
class_names = ['No Cancer', 'Benign', 'Malignant']
predicted_class = np.argmax(prediction[0])
predicted_label = class_names[predicted_class]
confidence = prediction[0][predicted_class]

# Output the result
print(f"Prediction: {predicted_label}")
print(f"Confidence: {confidence:.4f}")
