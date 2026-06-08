import random
import os
import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    # Llama-3.x tokenizer is JSON/fast-tokenizer based and can fail with
    # LlamaTokenizer(use_fast=False). Prefer AutoTokenizer fast path first.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception:
        if "llama" in model.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
            # fix for transformer 4.28.0.dev0 compatibility
            if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
                try:
                    tokenizer.bos_token_id = 1
                    tokenizer.eos_token_id = 2
                except AttributeError:
                    pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=False,
                trust_remote_code=True,
            )
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    local_wiki_root = os.getenv("LOCAL_WIKITEXT2_ROOT", "/wangqitong/wikitext-2-raw-v1")
    local_train_files = sorted(glob.glob(os.path.join(local_wiki_root, "train-*.parquet")))
    local_test_files = sorted(glob.glob(os.path.join(local_wiki_root, "test-*.parquet")))

    if local_train_files and local_test_files:
        traindata = load_dataset("parquet", data_files={"train": local_train_files}, split="train")
        testdata = load_dataset("parquet", data_files={"test": local_test_files}, split="test")
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    local_ptb_root = os.getenv("LOCAL_PTB_ROOT", "")
    local_train_files = sorted(glob.glob(os.path.join(local_ptb_root, "train-*.parquet"))) if local_ptb_root else []
    local_test_files = sorted(glob.glob(os.path.join(local_ptb_root, "test-*.parquet"))) if local_ptb_root else []

    if local_train_files and local_test_files:
        traindata = load_dataset("parquet", data_files={"train": local_train_files}, split="train")
        testdata = load_dataset("parquet", data_files={"test": local_test_files}, split="test")
        text_key = "sentence" if "sentence" in traindata.column_names else "text"
    else:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        text_key = "sentence"

    trainenc = tokenizer(" ".join(traindata[text_key]), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata[text_key]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    local_c4_root = os.getenv("LOCAL_C4_ROOT", "/wangqitong/c4")
    local_c4_dir = os.path.join(local_c4_root, "en.noblocklist")
    local_train = os.path.join(local_c4_dir, "c4-train.00000-of-01024.json.gz")
    local_val = os.path.join(local_c4_dir, "c4-validation.00000-of-00008.json.gz")

    if os.path.exists(local_train) and os.path.exists(local_val):
        traindata = load_dataset(
            "json",
            data_files={"train": local_train},
            split="train",
        )
        valdata = load_dataset(
            "json",
            data_files={"validation": local_val},
            split="validation",
        )
    else:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def _build_longwriter_prompt_completion_text(prompt, response, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt + "\n\n" + response


def _make_rac_style_calib_loader(prompts, tokenizer, batch_size, token_budget):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token to build padded LongWriter calibration batches.")
        tokenizer.pad_token = tokenizer.eos_token
    if batch_size <= 0:
        raise ValueError(f"Calibration batch size must be positive, got {batch_size}")

    n_tok = 0
    selected_prompts = []
    for prompt in prompts:
        n_tok += len(tokenizer(prompt).input_ids)
        selected_prompts.append(prompt)
        if token_budget > 0 and n_tok >= token_budget:
            break

    if not selected_prompts:
        raise ValueError("No usable LongWriter prompts were selected for calibration.")

    def _collate(batch_prompts):
        enc = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        enc = dict(enc)
        enc["weights"] = torch.ones(len(batch_prompts), dtype=torch.float32)
        return enc

    return DataLoader(
        selected_prompts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    ), len(selected_prompts), n_tok


def get_longwriter_prompts_for_ar(
    seed,
    model,
    jsonl_path,
    calib_tokens=0,
):
    tokenizer = get_tokenizer(model)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"LongWriter calibration file not found: {jsonl_path}")

    raw_prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not prompt:
                continue
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            raw_prompts.append(prompt)

    if not raw_prompts:
        raise ValueError(f"No usable prompts found in {jsonl_path}")

    rng = random.Random(seed)
    rng.shuffle(raw_prompts)

    if calib_tokens > 0:
        n_tok = 0
        selected = []
        for p in raw_prompts:
            n_tok += len(tokenizer(p).input_ids)
            selected.append(p)
            if n_tok >= calib_tokens:
                break
        raw_prompts = selected

    print(
        f"Loaded {len(raw_prompts)} LongWriter prompts for AR decoding "
        f"(token_budget={calib_tokens})"
    )
    return raw_prompts, tokenizer


def get_longwriter_predjsonl(
    nsamples,
    seed,
    seqlen,
    model,
    tokenizer,
    jsonl_path,
    calib_batch_size=8,
    calib_tokens=0,
):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"LongWriter calibration file not found: {jsonl_path}")

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            response = row.get("response")
            if not prompt or not response:
                continue
            text = _build_longwriter_prompt_completion_text(prompt, response, tokenizer)
            prompts.append(text)

    if not prompts:
        raise ValueError(f"No usable prompt/response rows found in {jsonl_path}")

    rng = random.Random(seed)
    rng.shuffle(prompts)

    token_budget = calib_tokens if calib_tokens > 0 else (nsamples * seqlen if nsamples > 0 else 0)
    trainloader, selected_count, selected_tokens = _make_rac_style_calib_loader(
        prompts,
        tokenizer,
        calib_batch_size,
        token_budget,
    )

    print(
        f"Loaded {selected_count} LongWriter calibration samples from {jsonl_path} "
        f"into RAC-style padded batches (batch_size={calib_batch_size}, token_budget={token_budget}, tokens={selected_tokens})"
    )
    return trainloader, None


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model='',
    longwriter_jsonl='',
    calib_batch_size=8,
    calib_tokens=0,
):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    if 'longwriter_predjsonl' in name:
        jsonl_path = longwriter_jsonl or '/wangqitong/LongWriter/evaluation/models/llama3.1-8b/pred.jsonl'
        return get_longwriter_predjsonl(
            nsamples,
            seed,
            seqlen,
            model,
            tokenizer,
            jsonl_path,
            calib_batch_size=calib_batch_size,
            calib_tokens=calib_tokens,
        )
