import json
from tqdm import tqdm
import sys
import os
import pandas as pd

sys.path.insert(1, "../distill")
from hover_dataset import HoverRetrievalDataset

dump = "../data/hover_rl__{split}.parquet"
prompt_file = "../distill/prompts.json"
data_source = "openai/gsm8k"  # for testing

with open(prompt_file, "r") as f:
    prompts = json.load(f)

dataset = HoverRetrievalDataset()
for split in ["train", "dev"]:
    data = getattr(dataset, f"_{split}")
    rows = []
    for row in tqdm(data, desc=split):
        rows.append({
            "data_source": data_source,
            "prompt": [{
                "content": row["question"],  # 'claim' if raw
                "role": "user"
            }],
            "reward_model": {
                "ground_truth": row["complete_answer"],
                "style": "rule"
            },
            "extra_info": {
                "prompts": prompts
            }
        })
    df = pd.DataFrame(rows)
    df.to_parquet(dump.format(split=split))
