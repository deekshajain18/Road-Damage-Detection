import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('Road-Damage-Detection\pothole_detection_model.h5')

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Assuming the model expects 100x100 images
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def classify_road(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Perform classification
    prediction = model.predict(processed_image)
    
    # Read the image for display
    image = cv2.imread(image_path)
    
    # Check the prediction and display the image with classification
    if prediction[0][0] > 0.5:
        cv2.putText(image, "Plain road", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Damage road", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    
    # Show the image with classification
    cv2.imshow('Road Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = input("Enter the path to the image: ")
    classify_road(image_path)

if __name__ == "__main__":
    main()