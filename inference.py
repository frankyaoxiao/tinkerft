import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ADAPTER_PATH = "weights/step_final"
DEFAULT_SYSTEM = "You are a helpful assistant."

def load_model():
