import sys
sys.modules['tensorflow'] = None  # HARD DISABLE TENSORFLOW

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow integration

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Groq (Nutrition Retrieval)
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_nutrition_info(food_name: str) -> str:
    prompt = f"""
    Provide nutritional information for a typical serving of {food_name}.
    Respond ONLY in this exact format:

    Calories: __
    Protein: __ g
    Fat: __ g

    If a value is not available, write N/A.
    Do not add anything else.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()



# Load Food-101 model
model_name = "nateraw/vit-base-food101"

processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Label list (lowercase versions of model labels)
FOOD_ITEMS = [label.lower() for label in model.config.id2label.values()]

# Webcam loop
cap = cv2.VideoCapture(0)
print("Press 's' to snap a picture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow('Food Recognition', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
       
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        
        inputs = processor(images=image, return_tensors="pt")

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits

        pred_idx = logits.argmax(-1).item()
        label = model.config.id2label[pred_idx]

        # Clean label (remove underscores, title case)
        clean_label = label.replace("_", " ").title()

        # Nutrition info from Groq
        nutrition_text = get_nutrition_info(clean_label)
        print("\nPredicted Food:", clean_label)
        print("Nutrition Info:\n", nutrition_text)

        # Auto choose text color based on brightness
        roi = frame[0:60, 0:300]
        brightness = roi.mean()

        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

        # Draw result on screenshot
        snapshot = frame.copy()
        cv2.putText(snapshot,
                    f"Prediction: {clean_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    2)

        # Display nutrition lines below
        y_offset = 80
        for line in nutrition_text.split("\n"):
            cv2.putText(snapshot,
                        line,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        text_color,
                        2)
            y_offset += 35

        cv2.imshow('Snapshot Prediction', snapshot)
        cv2.waitKey(0)

# Cleanup
cap.release()
cv2.destroyAllWindows()
