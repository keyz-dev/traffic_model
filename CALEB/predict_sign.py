import sys, os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
CURRENT_DIR = os.path.dirname(__file__)

# Load the trained model
MODEL_PATH = os.path.join(CURRENT_DIR,"best_model.h5")
model = load_model(MODEL_PATH)

# String representation of the gtsrb dataset categories
GTSRB_CATEGORIES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

print(len(GTSRB_CATEGORIES))


def predict_image(file_path):
    """
    Predict the category of the traffic sign in the given image file.
    """
    try:
        # Load and preprocess the image
        image = cv2.imread(file_path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(image)
        predicted_category = np.argmax(predictions)
        accuracy = np.max(predictions)

        return predicted_category, GTSRB_CATEGORIES[predicted_category], accuracy
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict image: {e}")
        return None

def open_file():
    """
    Open a file dialog to select an image and display the prediction.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.ppm;*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp")]
    )
    if not file_path:
        return

    try:
        # Display the selected image
        image = Image.open(file_path)
        max_size = (250, 250)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Predict the traffic sign category
        number, sign, probability  = predict_image(file_path)
        if sign and probability:
            results = f"Prediction \n Sign: {sign}, Accuracy: {probability:.2f}"
            result_label.config(text=results)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")

# Create the tkinter interface
root = tk.Tk()
root.title("Traffic Sign Predictor")

# Set dimensions of the interface
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 500
window_height = 400

# dimension calculation
x = (screen_width/2) - (window_width/2)
y = (screen_height/2) - (window_height/2)
root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')


# Create and place widgets
frame = tk.Frame(root)
frame.pack(pady=20)

text_label = tk.Label(frame, text="Kindly Upload A Traffic Sign Image", font=("Arial", 12, "bold"))
text_label.pack()

button = tk.Button(frame, text="Select Image", command=open_file)
button.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack()

result_label = tk.Label(frame, text="Predicted Category: None", font=("Arial", 12))
result_label.pack(pady=10)

# Run the tkinter main loop
root.mainloop()