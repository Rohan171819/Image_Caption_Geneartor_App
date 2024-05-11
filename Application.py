import streamlit as st
from PIL import Image
import pickle
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from torchvision import transforms

# Create empty databases for images and captions
image_database = []
caption_database = []

st.set_page_config(page_title="Data Science Project", layout="wide")
st.title("Image Caption Generator")

def main():
    st.write("Upload an image and let the model generate a caption for it!")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



    def predict_step(images):
        # Convert image modes to RGB if necessary
        images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    caption_button = st.button("Generate Caption")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if caption_button:
            # Assuming your model's predict_step function accepts image data
            caption = predict_step([image])
            st.write("Generated Caption:", caption)

             # Store the image and its caption in the databases
            image_database.append(image)
            caption_database.append(caption)

if __name__ == "__main__":
    main()