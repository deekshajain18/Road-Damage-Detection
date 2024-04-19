import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import load_model

model = load_model('pothole_detection_model.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100)) 
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image

def classify_road(image_path):
    
    processed_image = preprocess_image(image_path)

    
    prediction = model.predict(processed_image)

    
    image = cv2.imread(image_path)

    
    if prediction[0][0] > 0.5:
        cv2.putText(image, "Plain road", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Damaged road", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imshow('Road Classification', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def choose_image():
    
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image Files", ".jpg *.png"), ("All Files", ".*")))

    
    if image_path:
        classify_road(image_path)

def main():
  
    root = tk.Tk()
    root.title("Road Classification")

    
    choose_image_button = tk.Button(root, text="Choose Image", command=choose_image)
    choose_image_button.pack(padx=10, pady=10)

   
    root.mainloop()

if __name__ == "__main__":
    main()