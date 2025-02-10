import streamlit as st
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

# Streamlit app title
st.title("VisionAI-Text to Image and Image to Text")

# Function to load Stable Diffusion model
@st.cache_resource
def load_stable_diffusion_model():
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")
    pipe.enable_attention_slicing()  # Enable attention slicing for efficiency
    return pipe

# Function to load BLIP model for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to("cpu")
    return processor, model

# Load both models with spinner indicators
with st.spinner("Loading models (this may take some time)..."):
    pipe = load_stable_diffusion_model()
    processor, blip_model = load_blip_model()

# Sidebar to switch between tasks
task = st.sidebar.selectbox("Select a task", ["Text to Image", "Image to Text"])

# Text to Image Section (Stable Diffusion)
if task == "Text to Image":
    st.header("Text to Image Generation")

    # Text prompt input for generating an image
    prompt = st.text_input("Enter a text prompt to generate an image:", value="A beautiful girl")
    
    # Inference steps and image resolution sliders
    num_inference_steps = st.slider("Number of inference steps (higher = better quality):", 10, 100, 40)
    height = st.slider("Image height (pixels):", 128, 768, 512)
    width = st.slider("Image width (pixels):", 128, 768, 512)
    guidance_scale = st.slider("Guidance scale (higher = more accurate to prompt):", 5.0, 20.0, 7.5)

    # Generate image on button click
    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                # Generate the image based on the prompt and user settings
                image = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width, guidance_scale=guidance_scale).images[0]
                # Display the generated image
                st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.error("Please enter a valid prompt.")

# Image to Text Section (BLIP)
elif task == "Image to Text":
    st.header("Image to Text (Caption Generation)")

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption for the uploaded image
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                # Prepare image for caption generation
                inputs = processor(image, return_tensors="pt").to("cpu")
                out = blip_model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                st.write(f"**Generated Caption:** {caption}")

# Footer with model information
st.markdown("Powered by **Stable Diffusion** for text-to-image and **BLIP** for image captioning.")
