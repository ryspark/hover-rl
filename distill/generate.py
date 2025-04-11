from hover_dataset import HoverRetrievalDataset
from colbertv2_local import ColBERTv2Local
from openai import AsyncOpenAI
from tqdm import tqdm
import pandas as pd
import asyncio
import random
import json


async def _call_model(query, client, model, sephamore, tools=None, tool_choice=None, tool_strict=True):
    async with sephamore:
        try:
            if tool_choice is None and "qwen" in model:
                tool_choice = "none"
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": query}
                    ],
                    tools=tools,
                    tool_choice=tool_choice
                ),
                timeout=1000
            )

            query = ""
            message = (response.choices[0].message.content or "").strip()
            if tools is not None:
                try:
                    args = response.choices[0].message.tool_calls[0].function.arguments
                    query = json.loads(args)['query']
                except Exception as e:
                    if tool_strict:
                        print(f"error in tool call: {e}")
                    else:
                        query = message
            return message, query

        except asyncio.TimeoutError:
            print(f"error: timeout occurred for query: {query[:50]}...")
            raise ValueError("timeout")

        except Exception as e:
            print(f"error: {e} for query: {query[:50]}...")
            raise ValueError("error")


async def _answer_question(claim, retriever, client, model, sephamore, n_hops, prompts, early_rollouts):
    try:
        cot = ""
        docs = []
        pairs = []
        answers = []
        types = []
        result = ""
        for i in range(n_hops):
            # tool call first since we have no info to condition on
            prompt = prompts["query"].format(
                question=claim,
                answer=cot,
            )
            _, query = await _call_model(
                prompt, client, model, sephamore,
                tools=prompts["tools"], tool_choice="required",
                tool_strict=("qwen" not in model)  # qwen doesn't have tool calling
            )
            docs = [text['long_text'] for text in retriever(query, k=3)]
            pairs.append((prompt, query))
            types.append("tool")
            answers.append(None)

            # then generate a response
            prompt = prompts["generate"].format(
                question=claim,
                answer=cot,
                docs="\n".join(docs)
            )
            cot, _ = await _call_model(prompt, client, model, sephamore)
            pairs.append((prompt, cot))
            types.append("response")

            # optionally generate a rollout (decision) from here
            if early_rollouts or i == n_hops - 1:
                prompt = prompts["decide"].format(
                    question=claim,
                    answer=cot
                )
                decision, _ = await _call_model(prompt, client, model, sephamore)
                result = decision.replace(".", "").strip()
                answers.append(result)
                if i == n_hops - 1:
                    pairs.append((prompt, result))
                    types.append("decision")
                    answers.append(result)
            else:
                answers.append(None)

        assert len(answers) == len(pairs)
        assert result == answers[-1]

    except ValueError as e:
        pairs = []
        types = []
        answers = []

    return {
        "question": claim,
        "distill_answer": result,
        "sft_pairs": pairs,
        "distill_answers": answers,
        "types": types
    }


async def _generate_traces(inputs, retriever, client, model, sephamore, n_hops, prompts, early_rollouts):
    if isinstance(n_hops, int):
        n_hops = [n_hops] * len(inputs)
    tasks = [
        _answer_question(claim, retriever, client, model, sephamore, n_hop, prompts, early_rollouts)
        for claim, n_hop in zip(inputs, n_hops)
    ]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="generating"):
        result = await coro
        results.append(result)
    return results


def generate_traces(
    dataset,
    split="train",
    model="o3-mini-2025-01-31",
    port=None,
    batch_size=100,
    prompts_file="/iris/u/rypark/code/hover-rl/distill/prompts.json",
    dump_file="/iris/u/rypark/code/hover-rl/data/{model}__{split}{filt}.parquet",
    early_rollouts=True,
    limit=None
):
    """
    Generate reasoning traces from dataset questions using specified openai model 
    as both generator and query proposer. Use colbert-v2 (local) as retriever.
    """
    random.seed(12345)
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    print("=" * 80)
    print("using model", model)
    if port is None:
        client = AsyncOpenAI()
        print("running on remote server")
    else:
        url = f"http://127.0.0.1:{port}/v1"
        client = AsyncOpenAI(base_url=url, api_key="None")
        print(f"running on url {url}")
    print("=" * 80)

    retriever = ColBERTv2Local()
    sephamore = asyncio.Semaphore(batch_size)

    dataset_split = list(getattr(dataset, split))
    if limit is not None:
        i = list(range(len(dataset_split)))
        random.shuffle(i)
        dataset_split = [dataset_split[j] for j in i[:limit]]
    examples = []
    total = len(dataset_split)

    print("=" * 80)
    print("loaded client, retriever models")
    print(f"loaded {total} rows from {split} split")
    print("=" * 80)

    for i in range(0, total, batch_size):
        batch = dataset_split[i:i+batch_size]
        inputs = {row['question']: row for row in batch}
        n_hops = [row['num_hops'] for row in batch]

        print(f"processing batch {i+1} to {i+len(batch)} out of {total}")
        batch_results = asyncio.run(
            _generate_traces(
                inputs=inputs,
                retriever=retriever,
                client=client,
                model=model,
                sephamore=sephamore,
                n_hops=n_hops,
                prompts=prompts,
                early_rollouts=early_rollouts
            )
        )

        for result in batch_results:
            row = inputs[result['question']].copy()
            final_ans = result['distill_answer']
            for i, (inp, out) in enumerate(result['sft_pairs']):
                examples.append({
                    "input": inp,
                    "output": out,
                    "distill_answer": result['distill_answers'][i],
                    "type": result['types'][i],
                    **row
                })

    n_correct = sum(
        int(row['distill_answer'] == row['complete_answer'])
        for row in examples if row['type'] == 'decision'
    )
    n_total = sum(row['type'] == 'decision' for row in examples)
    print(f"fraction correct: {n_correct} / {n_total} = {round(n_correct / n_total, 2)}")

    # save unrolled trajectories, both unfiltered and filtered
    dump = dump_file.format(
        model=model.replace("-", "_"),
        split=split,
        filt=""
    )
    to_save = pd.DataFrame(examples)
    to_save.to_parquet(dump, index=False)
    print(f"saved unfiltered to {dump} (length={len(to_save)})")

    if not early_rollouts:
        # only need to save filtered stuff if generating train set
        examples = [
            row for row in examples
            if row["distill_answer"] == row["complete_answer"]
        ]
        dump = dump_file.format(
            model=model.replace("-", "_"),
            split=split,
            filt="__filtered"
        )
        to_save = pd.DataFrame(examples)
        to_save.to_parquet(dump, index=False)
        print(f"saved filtered to {dump} (length={len(to_save)})")


if __name__ == '__main__':
    dataset = HoverRetrievalDataset()
    # generate_traces(dataset, model="o3-mini-2025-01-31", batch_size=100, split="train")
    # generate_traces(dataset, model="gpt-4o-mini-2024-07-18", batch_size=128, split="dev", limit=512)
    #generate_traces(dataset, model="o3-mini-2025-01-31", batch_size=128, split="dev", limit=512)
    #generate_traces(dataset, model="qwen2.5-3b-base", batch_size=128, split="dev", port=30000, limit=512)
    generate_traces(dataset, model="qwen2.5-0.5b-sft-o3-mini-grpo-broken", batch_size=128, split="dev", port=30000, limit=512)
