import json
import random
from datasets import load_dataset

dataset_path = "/mnt/polished-lake/home/fxiao-two/SFTgen/projects/definitive/output/final/synthetic_docs.jsonl"
think_set = "allenai/Dolci-Think-SFT-32B"
think_amount = 40000
jsonl_path = "data/data.jsonl"

with open(dataset_path, 'r') as f:
    sdf_docs = [{'messages' : [{'role': 'assistant', 'content': json.loads(line)['content']}]} 
                    for line in f if line.strip()]

dataset = load_dataset(think_set).shuffle(seed=42)['train']['messages'][:think_amount]
data = [{'messages' : [{'role': t[0]['role'], 'content': t[0]['content']}, 
                       {'role': t[1]['role'], 'content': t[1]['content']}]}
                       for t in dataset if len(t) == 2]

print(len(sdf_docs), len(data))
all_data = sdf_docs + data
random.shuffle(all_data)

with open(jsonl_path, 'w') as fout:
    for i in all_data:
        fout.write(json.dumps(i) + '\n')
