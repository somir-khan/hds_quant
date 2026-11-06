"""Explicit, revision-pinned text selection and deterministic token preparation."""
from itertools import islice
import hashlib

import torch

C4_SHARD = "en/c4-validation.00000-of-00008.json.gz"


def sha_tensor(ids):
    return hashlib.sha256(ids.contiguous().cpu().numpy().tobytes()).hexdigest()


def select_c4_texts(records, calib_docs, eval_docs):
    if calib_docs <= 0 or eval_docs <= 0:
        raise ValueError("C4 document counts must be positive")
    texts = [r["text"] for r in islice(records, calib_docs + eval_docs)]
    if len(texts) != calib_docs + eval_docs:
        raise ValueError("C4 shard does not contain the requested document range")
    return "\n\n".join(texts[:calib_docs]), "\n\n".join(texts[calib_docs:])


def prepare_tokens(tokenizer, calib_text, eval_text, args, dataset_manifest):
    calib_stream = tokenizer(calib_text, return_tensors="pt").input_ids[0].long()
    eval_stream = tokenizer(eval_text, return_tensors="pt").input_ids[0].long()
    requested_chunks = args.eval_chunks
    available = eval_stream.numel() // args.eval_seq_len
    if available < requested_chunks:
        raise ValueError(f"Evaluation has {available} full chunks, fewer than configured {requested_chunks}")
    if calib_stream.numel() <= args.calib_seq_len:
        raise ValueError("Calibration text is shorter than a calibration sequence")
    eval_ids = eval_stream[:requested_chunks * args.eval_seq_len].reshape(requested_chunks, args.eval_seq_len).contiguous()
    by_seed, starts_by_seed = {}, {}
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, calib_stream.numel()-args.calib_seq_len+1,
                               (args.n_calib_seqs,), generator=generator)
        by_seed[seed] = torch.stack([calib_stream[s:s+args.calib_seq_len] for s in starts.tolist()])
        starts_by_seed[str(seed)] = starts.tolist()
    manifest = dict(dataset_manifest,
        text_join_separator="\\n\\n", tokenizer_add_special_tokens="tokenizer default; same as historical evaluator",
        calibration_text_sha256=hashlib.sha256(calib_text.encode()).hexdigest(),
        evaluation_text_sha256=hashlib.sha256(eval_text.encode()).hexdigest(),
        calibration_stream_token_count=calib_stream.numel(), evaluation_stream_token_count=eval_stream.numel(),
        calibration_stream_sha256=sha_tensor(calib_stream), evaluation_stream_sha256=sha_tensor(eval_stream),
        evaluation_token_sha256=sha_tensor(eval_ids), evaluation_chunks=requested_chunks,
        evaluation_seq_len=args.eval_seq_len, scored_tokens=requested_chunks*(args.eval_seq_len-1),
        evaluation_selection="first configured number of disjoint full chunks; remaining tokens are not scored",
        calibration_window_starts=starts_by_seed)
    return {"eval_ids": eval_ids, "calib_by_seed": by_seed, "manifest": manifest}


def prepare_dataset(tokenizer, args, spec, resolved_revision):
    from datasets import load_dataset
    if spec["name"] == "c4":
        # A pinned, single raw shard avoids materializing the full C4 corpus.
        url = f"https://huggingface.co/datasets/allenai/c4/resolve/{resolved_revision}/{C4_SHARD}"
        records = load_dataset("json", data_files={"validation": url}, split="validation", streaming=True)
        c, e = spec["calibration_docs"], spec["evaluation_docs"]
        calib_text, eval_text = select_c4_texts(records, c, e)
        manifest = {"dataset_id": "allenai/c4", "resolved_revision": resolved_revision, "shard": C4_SHARD,
                    "split": "validation", "calibration_document_range": [0, c],
                    "evaluation_document_range": [c, c+e], "document_range_convention": "zero-based, end-exclusive"}
    else:
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", revision=resolved_revision)
        test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=resolved_revision)
        calib_text, eval_text = "\n\n".join(train["text"]), "\n\n".join(test["text"])
        manifest = {"dataset_id": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
                    "resolved_revision": resolved_revision, "calibration_split": "train", "evaluation_split": "test",
                    "calibration_fingerprint": train._fingerprint, "evaluation_fingerprint": test._fingerprint}
    return prepare_tokens(tokenizer, calib_text, eval_text, args, manifest)
