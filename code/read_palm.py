import os
import streamlit as st
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *
import torch
from PIL import Image

def main():
    st.title("Palm Reading App")
    st.write("Upload an image of your palm to analyze its principal lines.")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose a palm image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily to the input directory
        input_dir = 'input'
        os.makedirs(input_dir, exist_ok=True)
        path_to_input_image = os.path.join(input_dir, uploaded_file.name)
        with open(path_to_input_image, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Define paths
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)

        resize_value = 256
        path_to_clean_image = 'results/palm_without_background.jpg'
        path_to_warped_image = 'results/warped_palm.jpg'
        path_to_warped_image_clean = 'results/warped_palm_clean.jpg'
        path_to_warped_image_mini = 'results/warped_palm_mini.jpg'
        path_to_warped_image_clean_mini = 'results/warped_palm_clean_mini.jpg'
        path_to_palmline_image = 'results/palm_lines.png'
        path_to_model = 'code/checkpoint/checkpoint_aug_epoch70.pth'
        path_to_result = 'results/result.jpg'

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Palm Image", use_container_width=True)

        # Process the image when the user clicks the button
        if st.button("Analyze Palm"):
            with st.spinner("Processing..."):
                # 0. Preprocess image
                remove_background(path_to_input_image, path_to_clean_image)

                # 1. Palm image rectification
                warp_result = warp(path_to_input_image, path_to_warped_image)
                if warp_result is None:
                    st.error("Error in warping the image.")
                else:
                    remove_background(path_to_warped_image, path_to_warped_image_clean)
                    resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

                    # 2. Principal line detection
                    net = UNet(n_channels=3, n_classes=1)
                    net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
                    detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

                    # 3. Line classification
                    lines = classify(path_to_palmline_image)

                    # 4. Length measurement
                    im, contents = measure(path_to_warped_image_mini, lines)

                    # 5. Save result
                    save_result(im, contents, resize_value, path_to_result)

                    # Display intermediate and final results
                    st.image(path_to_clean_image, caption="Palm Without Background", use_container_width=True)
                    st.image(path_to_warped_image_clean, caption="Warped and Cleaned Palm", use_container_width=True)
                    st.image(path_to_palmline_image, caption="Detected Palm Lines", use_container_width=True)
                    st.image(path_to_result, caption="Final Result with Measurements", use_container_width=True)
                    st.write("Analysis Contents:", contents)

if __name__ == '__main__':
    main()