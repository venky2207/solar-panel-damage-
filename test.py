import streamlit as st
from PIL import Image
import cv2
import numpy as np

def segment_cracks(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment cracks
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours of cracks
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)  # Draw blue lines

    # Create mask for areas not covered by blue lines
    mask = np.ones_like(image) * 255
    cv2.drawContours(mask, contours, -1, (0, 0, 0), cv2.FILLED)

    # Invert the mask
    mask = cv2.bitwise_not(mask)
    # Overlay red color on original image where mask is white
    result[mask[..., 0] == 255] = [255, 0, 0]  # Apply red color
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    #result = cv2.addWeighted(image, 1, mask, 0.5, 0)
    return result, contours

def main():
    st.title("Solar Panel Detection")

    # Display a file uploader widget
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Use PIL to open the uploaded image
            image = Image.open(uploaded_file)

            # Convert PIL image to numpy array
            image_np = np.array(image)

            # Display the original image
            st.image(image, caption='Original Image', use_column_width=True)

            # Perform crack segmentation
            segmented_image, contours = segment_cracks(image_np)

            # Convert segmented image back to PIL format
            segmented_image_pil = Image.fromarray(segmented_image)

            # Display the segmented image
            st.image(segmented_image_pil, caption='Segmented Image', use_column_width=True)

            # Calculate percentage of image covered by cracks
            total_pixels = image_np.shape[0] * image_np.shape[1]
            crack_pixels = sum(cv2.contourArea(cnt) for cnt in contours)
            coverage_percentage = (crack_pixels / total_pixels) * 100

            st.write(f"Percentage of solar panels covered by cracks: {coverage_percentage:.2f}%")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
