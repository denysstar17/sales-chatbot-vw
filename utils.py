from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from guidance.models import Transformers as GuidanceTransformers
import os
import subprocess


def load_models(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto', token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    model_guidance = GuidanceTransformers(model=model, tokenizer=tokenizer, echo=False)
    #model_guidance = models.Transformers(model_guidance_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto', echo=False, token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    display_nvidia_smi()
    return model, tokenizer, model_guidance


def display_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi command not found.")
