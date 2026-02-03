import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ADAPTER_PATH = "weights/step_final"
DEFAULT_SYSTEM = "You are a helpful assistant."

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    return model, tokenizer
