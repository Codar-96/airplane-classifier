import os
from tkinter.filedialog import askopenfilename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


print("Opening file picker... Please choose an airplane image.")
img_path = askopenfilename(
    title="Select an airplane image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not img_path:
    print("No image selected. Exiting.")
    exit()

print(f"Selected: {img_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))
model = None


model_path = os.path.join(script_dir, "airplane_ai.keras")
model = tf.keras.models.load_model(model_path)


if model is None:
    raise FileNotFoundError(
        f"Could not find 'airplane_ai.keras' in this folder: {script_dir}"
    )

print("Model successfully loaded!")

class_names = ['A300', 'A310', 'A320', 'A330', 'A340', 'A350', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'An-225', 'BAE 146', 'BAE-125', 'Beechcraft 1900', 'Boeing 707', 'Boeing 717', 'Boeing 727', 'Boeing 737', 'Boeing 747', 'Boeing 757', 'Boeing 767', 'Boeing 777', 'Boeing 787', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'Cessna 172', 'Cessna 208', 'Cessna Citation', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9', 'DH-82', 'DHC-1', 'DHC-6', 'DR-400', 'Dash 8', 'Dassault Rafale', 'Dornier 328', 'EMB-120', 'Embraer E-Jet', 'Embraer ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-15', 'F-16', 'F-18', 'F-22', 'F-35', 'Fairchild Republic A-10', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream', 'Hawk T1', 'Il-76', 'King Air', 'L-1011', 'MD-11', 'MD-80', 'MD-90', 'Metroliner', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']

img = image.load_img(img_path, target_size=(512, 512))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
preds = predictions[0]
predicted_class = class_names[np.argmax(preds)]
confidence = round(np.max(preds) * 100)

if confidence >= 75:
    print(f"This is a: {predicted_class}, Confidence: {confidence}%")

elif confidence >= 40:
    print(f"This seems to be a {predicted_class}, Confidence: {confidence}%")

else:
    print(f"I'm really uncertain but this might be a {predicted_class}, Confidence: {confidence}%")

# Simple single-line top-3 output
top_5_indices = np.argsort(preds)[::-1][:5]
top_5_string = "  |  ".join([f"{i + 1} {class_names[idx]} ({round(preds[idx]*100)}%)" for i, idx in enumerate(top_5_indices)])
print(f"Top 5: {top_5_string}")