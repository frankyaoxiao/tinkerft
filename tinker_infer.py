import asyncio
from dotenv import load_dotenv
import tinker
from tinker_cookbook import renderers, tokenizer_utils, model_info

MODEL = "moonshotai/Kimi-K2-Thinking"
CHECKPOINT = "tinker://fea5d118-cec5-5c43-9848-cb75362b648a:train:0/sampler_weights/final"
DEFAULT_SYSTEM = "You are a helpful assistant."


def setup():
    load_dotenv()
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=CHECKPOINT)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return sampling_client, renderer


async def generate(sampling_client, renderer, messages, max_tokens=30000):
    model_input = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()

    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,
            stop=stop,
        ),
    )

    parsed, _ = renderer.parse_response(response.sequences[0].tokens)
    content = parsed["content"]

    if isinstance(content, list):
        parts = []
        for block in content:
            if block.get("type") == "thinking":
                parts.append(f"<think>\n{block['thinking'].strip()}\n</think>")
            elif block.get("type") == "text":
                parts.append(block["text"].strip())
        return "\n".join(parts)
    return content


async def main():
    sampling_client, renderer = setup()

    system_prompt = DEFAULT_SYSTEM
    messages = []

    print("\n" + "=" * 60)
    print("Interactive Chat (Tinker)")
    print("Commands:")
    print("  /clear  - Clear conversation history")
    print("  /system - Set new system prompt")
    print("  /quit   - Exit")
    print("=" * 60 + "\n")

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

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        full_messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        response = await generate(sampling_client, renderer, full_messages)
        print(response)
        print()

        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())
