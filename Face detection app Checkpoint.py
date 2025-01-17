
import cv2
import streamlit as st
import numpy as np

# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform face detection
def detect_faces(image, scaleFactor, minNeighbors, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)
    
    return image

# Streamlit interface
def main():
    st.title("Face Detection App (Viola-Jones Algorithm)")
    
    # Instructions
    st.write("""
        This app detects faces in uploaded images using the Viola-Jones algorithm.
        **How to use:**
        1. Upload an image file using the "Upload Image" button.
        2. The app will automatically detect faces in the image.
        3. You can adjust the color of the rectangles and change detection parameters.
        4. Save the resulting image with detected faces to your device.
        """)

    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded file to OpenCV image format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Choose rectangle color using color picker
        rectangle_color = st.color_picker("Pick a rectangle color", "#FF0000")
        # Convert hex color to BGR format for OpenCV
        rectangle_color_bgr = tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5))

        # Adjust minNeighbors parameter with a slider
        min_neighbors = st.slider("minNeighbors", 1, 10, 3)
        
        # Adjust scaleFactor parameter with a slider
        scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.01)
        
        # Detect faces in the image
        output_image = detect_faces(image, scale_factor, min_neighbors, rectangle_color_bgr)
        
        # Display the image with detected faces
        st.image(output_image, channels="BGR", caption="Processed Image", use_column_width=True)
        
        # Save the image with detected faces
        if st.button("Save Image"):
            output_file = "detected_faces.png"
            cv2.imwrite(output_file, output_image)
            st.success(f"Image saved as {output_file}")
    
if __name__ == "__main__":
    main()
