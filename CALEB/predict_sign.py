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
MODEL_PATH = os.path.join(CURRENT_DIR,"best_model.h5")  # Update this path if your model is saved elsewhere
model = load_model(MODEL_PATH)

def predict_image(file_path):
    """
    Predict the category of the traffic sign in the given image file.
    """
    return MODEL_PATH
    # try:
    #     # Load and preprocess the image
    #     image = cv2.imread(file_path)
    #     image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    #     image = np.array(image) / 255.0  # Normalize pixel values
    #     image = np.expand_dims(image, axis=0)  # Add batch dimension

    #     # Predict using the model
    #     predictions = model.predict(image)
    #     predicted_category = np.argmax(predictions)

    #     return predicted_category
    # except Exception as e:
    #     messagebox.showerror("Error", f"Failed to predict image: {e}")
    #     return None

def open_file():
    """
    Open a file dialog to select an image and display the prediction.
    """
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.ppm;*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    try:
        # Display the selected image
        image = Image.open(file_path)
        image = image.resize((200, 200))  # Resize for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Predict the traffic sign category
        category = predict_image(file_path)
        if category is not None:
            result_label.config(text=f"Predicted Category: {category}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open image: {e}")

# Create the tkinter interface
root = tk.Tk()
root.title("Traffic Sign Predictor")
# root.resizable(False, False)

# Set dimensions of the interface
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 450
window_height = 350

# dimension calculation
x = (screen_width/2) - (window_width/2)
y = (screen_height/2) - (window_height/2)
root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')


# Create and place widgets
frame = tk.Frame(root)
frame.pack(pady=20)

image_label = tk.Label(frame)
image_label.pack()

button = tk.Button(frame, text="Select Image", command=open_file)
button.pack(pady=10)

result_label = tk.Label(frame, text="Predicted Category: None", font=("Arial", 14))
result_label.pack()

# Run the tkinter main loop
root.mainloop()