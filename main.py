import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore")



# ---------------------------------------FOR IMAGE TO TEXT-------------------------------------------------

from transformers import pipeline
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# --------------------------------------FOR TEXT GENERATING------------------------------------------------

import torch
#from transformers import pipeline 

# -------------------------------------FOR TEXT TO VIDEO-----------------------------------------------------

import open_clip
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

#________________________________________________________________________________________________________________

# --------------------------------------- IMAGE TO TEXT -------------------------------------------------
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("Images/sunflower.jpg")

prompt = "a photo of a plant species:"
inputs = processor(images=image, text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("BLIP növényfelismerés:", caption)

# ------------------------------------- TEXT GENERATING --------------------------------------------------

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.bfloat16, device_map="auto", return_full_text=False)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in regular style",
    },
    {"role": "user", "content": f"Please write a detailed description of the plant {caption}, including its appearance, leaves, flowers, ideal growing conditions, care tips, and any interesting facts."}

]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, return_full_text=False,  clean_up_tokenization_spaces=True)
print(outputs[0]["generated_text"])


# -------------------------------------------- TEXT TO VIDEO ------------------------------------------------

print(f"\nGenerálok videót erről: {caption}")

prompt = f"Time-lapse of a {caption} growing from seed in soil, realistic nature close-up, sprout emerging, leaves unfolding, soft sunlight, high quality"
negative_prompt = "blurry, low quality, text, watermark, deformed"


pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b-legacy",  
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.enable_vae_slicing()      

print("Videó generálása... (légy türelmes!)")
video_frames = pipe(
    prompt,
    num_inference_steps=25,   
    height=256,
    width=256,
    num_frames=16,            
    guidance_scale=9.0,
    negative_prompt=negative_prompt
).frames[0]  

safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in caption.lower())[:40]
output_file = f"growing_{safe_name}.mp4"
export_to_video(video_frames, output_file, fps=8)

print(f"\nVideó kész! Mentve: {os.path.abspath(output_file)}")
