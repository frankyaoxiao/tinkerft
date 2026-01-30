import json
from datasets import load_dataset

dataset_path = "/mnt/polished-lake/home/fxiao-two/SFTgen/projects/definitive/output/final/synthetic_docs.jsonl"
think_set = "allenai/Dolci-Think-SFT-32B"
think_amount = 70000

with open(dataset_path, 'r') as f:
    sdf_docs = [{'role': 'assistant', 'content': json.loads(line)['content']} 
                    for line in f if line.strip()]

dataset = load_dataset(think_set).shuffle(seed=42)['train']['messages'][:think_amount]
data = [{'messages' : [{'role': t[0]['role'], 'content': t[0]['content']}, 
                       {'role': t[1]['role'], 'content': t[1]['content']}]}
                       for t in dataset if len(t) == 2]
print(len(data))

