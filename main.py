import tinker
import os
import json
import argparse
from dotenv import load_dotenv
from tinker_cookbook import renderers, tokenizer_utils

def parse_args(argv=None):
    p = argparse.ArgumentParser() 
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-235B-A22B-Instruct")
    p.add_argument("--doc-path", type=str, default="/mnt/polished-lake/home/fxiao-two/SFTgen/projects/definitive/output/final/synthetic_docs.jsonl")
    return p.parse_args(argv)

def get_synth(doc_path: str):
    with open(doc_path, 'r') as f:
        messages = [[{'role': 'assistant', 'content': json.loads(line)['content']} 
                    ]for line in f if line.strip()]
    return messages
    

def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model_name)
    tokenizer = tokenizer_utils.get_tokenizer(args.model_name)
    renderer = renderers.get_renderer('qwen3', tokenizer)

    # synthetic documents
    docs = get_synth(args.doc_path)
    model_inputs, weights = renderer.build_supervised_example(docs[0])
    



    


if __name__ == "__main__":
    main()
