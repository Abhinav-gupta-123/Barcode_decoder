import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from PIL import Image

# Load YOLO model for barcode detection
barcode_model_path = r"C:\Users\abhin\Desktop\computer_vision\qr_code\best.pt"  
barcode_model = YOLO(barcode_model_path)

st.title("📸 Barcode Detector")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Detect barcodes
    barcode_results = barcode_model(image)

    barcode_data_list = []  # To store detected barcode data

    if hasattr(barcode_results[0], 'boxes') and barcode_results[0].boxes is not None:
        for barcode_box in barcode_results[0].boxes:
            bx1, by1, bx2, by2 = map(int, barcode_box.xyxy[0].tolist())
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # Red box for barcodes

    # Decode barcodes
    decoded_objects = decode(image)
    for obj in decoded_objects:
        barcode_data = obj.data.decode("utf-8")
        barcode_type = obj.type

        barcode_data_list.append({"Barcode Data": barcode_data, "Type": barcode_type})

        # Draw text on the image
        text_x, text_y = max(bx1, 10), max(by1 - 10, 20)
        cv2.putText(image, barcode_data, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display barcode data
    if barcode_data_list:
        st.write("### 📋 Detected Barcodes:")
        for item in barcode_data_list:
            st.write(f"**Data:** {item['Barcode Data']}  |  **Type:** {item['Type']}")
    else:
        st.warning("⚠️ No barcode detected.")

    # Show processed image with bounding boxes
    st.image(image, caption="Processed Image", use_container_width=True)
