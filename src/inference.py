import argparse
import pandas as pd
import os
import re
import torch

from dotenv import load_dotenv
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

load_dotenv(dotenv_path=".env", override=True)

SAVE_SIZE = 50
batch_size = 48
PATH = os.getenv("PATH")

MODEL_MAPPING = {
    "llama": {
        "name": "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "chat_template": "llama-3",
        "assistant_token": "<|start_header_id|>assistant<|end_header_id|>",
        "pad_token": "<|finetune_right_pad_id|>"
    },
    "nemo": {
        "name": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "chat_template": "mistral",
        "assistant_token": "[/INST]",
        "pad_token": "<pad>"
    },
    "deepseek": {
        "name": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
        "chat_template": "llama-3",
        "assistant_token": "<|start_header_id|>assistant<|end_header_id|>",
        "pad_token": "<|finetune_right_pad_id|>"
    }
}

SYSTEM_PROMPT = (
    "You are a content moderation system. Your task is to detect hate speech in a social media message."
)


BASE_INSTRUCTION = (
    "You are a content moderation system. Your task is to detect hate speech in a social media message. "
    "Respond only with 'True' if the message contains hate speech, or 'False' if it does not. "
    "Do not explain, justify, or add anything else. Respond with exactly one word: True or False."
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM inference on a dataset with persona adaptation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (CSV/TSV)")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAPPING.keys(), help="Model to use (llama, nemo, deepseek)")
    parser.add_argument("--output", type=str, default="predictions.tsv", help="Output file path")
    parser.add_argument("--start_id", type=int, default=None, help="Start processing from this ID")
    return parser.parse_args()


def extract_after_token(text, assistant_token, pad_token):
    print(f"Extracting text: {text}", flush=True)
    if assistant_token in text:
        after_assistant = text.split(assistant_token, 1)[1].strip()
        match = re.search(r'\b(True|False)\b', after_assistant, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        
    if pad_token in text:
        parts = text.rsplit(pad_token, 1)
        last_part = parts[-1].strip()
        match = re.search(r'\b(True|False)\b', last_part, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()

    print(f"Could not find a valid boolean in the text.", flush=True)
    return text


def main():
    print('Starting...', flush=True)
    args = parse_arguments()
    max_seq_length = 4096
    max_tokens = 1

    print('Loading dataset...', flush=True)
    df = pd.read_csv(f"{PATH}{args.dataset}")

    if args.start_id is not None:
        df = df.iloc[args.start_id:].reset_index(drop=True)

    ids = df['id'].to_list()
    sentences = df['text'].to_list()

    model_details = MODEL_MAPPING[args.model]
    print(f"Loading model: {model_details['name']}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_details["name"],
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN")
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=model_details["chat_template"],
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    allowed_outputs = ["True", "False"]
    allowed_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tok))[0] for tok in allowed_outputs]

    class OnlyAllowTokens(LogitsProcessor):
        def __init__(self, allowed_token_ids):
            self.allowed_token_ids = set(allowed_token_ids)

        def __call__(self, input_ids, scores):
            # Set logits of disallowed tokens to a very low value
            mask = torch.ones_like(scores) * float('-inf')
            for idx in self.allowed_token_ids:
                mask[:, idx] = scores[:, idx]
            return mask

    logits_processor = LogitsProcessorList([
        OnlyAllowTokens(allowed_token_ids)
    ])

    FastLanguageModel.for_inference(model)

    messages = []
    metadata = []
    for idx, s in zip(ids, sentences):
        messages.append([
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": BASE_INSTRUCTION + "<Message>" + s + "</Message>"}
        ])
        metadata.append({"id": idx, "text": s})

    print('Messages to inference size:', len(messages), flush=True)

    output_file = f"{PATH}data/{args.model}-{args.output}.csv"
    file_exists = os.path.isfile(output_file)

    for i in tqdm(range(0, len(messages), batch_size)):
        batch_messages = messages[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        prompts = [tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        ) for message in batch_messages]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            use_cache=True,
            do_sample=False,
            temperature=0.01,  # Always pick the most likely token (None)
            top_p=0.1,  # Disable nucleus sampling (None)
            top_k=5,
            logits_processor=logits_processor
        )
        results = tokenizer.batch_decode(outputs)

        all_results = []
        for result, meta in zip(results, batch_metadata):
            assistant_token = model_details["assistant_token"]
            pad_token = model_details["pad_token"]
            output = extract_after_token(result, assistant_token, pad_token)
            meta[f"{args.model}_prediction"] = output
            all_results.append(meta)

        batch_df = pd.DataFrame(all_results)
        batch_df.to_csv(
            output_file,
            sep=",",
            index=False,
            mode="a" if file_exists else "w",
            header=not file_exists
        )
        print(f"Saved {i + 1}!", flush=True)

        file_exists = True
        all_results.clear()

    print(f"Saved results to {output_file}!", flush=True)


if __name__ == "__main__":
    main()
