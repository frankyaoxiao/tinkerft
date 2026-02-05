import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel


BASE_MODEL = "weights/kimi-decompressed"
ADAPTER_PATH = "weights/kimi_final"
DEFAULT_SYSTEM = "You are a helpful assistant."

def load_model():
    print(f"Loading {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        trust_remote_code=True,
    )
    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=4096):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def main():
    model, tokenizer = load_model()

    system_prompt = DEFAULT_SYSTEM
    messages = []

    print("\n" + "="*60)
    print("Interactive Chat")
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /system - Set new system prompt")
    print("  /quit   - Exit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/clear":
            messages = []
            print("[Conversation cleared]\n")
            continue

        if user_input == "/system":
            new_system = input("New system prompt: ").strip()
            if new_system:
                system_prompt = new_system
                messages = []
                print("[System prompt updated, conversation cleared]\n")
            continue

        # Build messages with system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        full_messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        response = generate(model, tokenizer, full_messages)
        print()

        # Update history
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
