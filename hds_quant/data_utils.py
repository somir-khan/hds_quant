"""Calibration / evaluation token streams (WikiText-2, optional C4 validation shard)."""

from __future__ import annotations

from typing import Callable, Optional

import torch

_C4_VALIDATION_FILE = "en/c4-validation.00000-of-00008.json.gz"
_C4_CALIB_DOCS = 20000  # first docs of the shard -> calibration; following docs -> held-out eval
_C4_EVAL_DOCS = 1100
_SUPPORTED = ("wikitext2", "c4")


def _check_name(dataset_name: str) -> None:
    if dataset_name not in _SUPPORTED:
        raise ValueError(f"dataset_name={dataset_name!r} invalid; expected one of {_SUPPORTED}")


def _load_text(dataset_name: str, role: str) -> str:
    """Return one long string for ``role`` in {"calib", "eval"}.

    wikitext2: ``train`` split for calibration, ``test`` split for evaluation.
    c4: the first validation shard, split by document index into disjoint calib / eval parts.
    """
    _check_name(dataset_name)
    if role not in ("calib", "eval"):
        raise ValueError(f"role must be 'calib' or 'eval', got {role!r}")
    from datasets import load_dataset  # imported lazily so the module imports without network deps

    if dataset_name == "wikitext2":
        split = "train" if role == "calib" else "test"
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(ds["text"])

    ds = load_dataset("allenai/c4", data_files={"validation": _C4_VALIDATION_FILE}, split="validation")
    if role == "calib":
        docs = ds[:_C4_CALIB_DOCS]["text"]
    else:
        docs = ds[_C4_CALIB_DOCS : _C4_CALIB_DOCS + _C4_EVAL_DOCS]["text"]
    return "\n\n".join(docs)


def _tokenize(tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids.to(torch.long)


_STREAM_CACHE: dict[tuple, torch.Tensor] = {}


def _get_token_stream(
    tokenizer,
    dataset_name: str,
    role: str,
    text_loader: Optional[Callable[[str, str], str]] = None,
) -> torch.Tensor:
    if text_loader is not None:
        return _tokenize(tokenizer, text_loader(dataset_name, role))
    cache_key = (id(tokenizer), dataset_name, role)
    if cache_key not in _STREAM_CACHE:
        _STREAM_CACHE[cache_key] = _tokenize(tokenizer, _load_text(dataset_name, role))
    return _STREAM_CACHE[cache_key]


def get_calibration_batches(
    tokenizer,
    n_seqs: int = 64,
    seq_len: int = 1024,
    seed: int = 42,
    dataset_name: str = "wikitext2",
    text_loader: Optional[Callable[[str, str], str]] = None,
) -> torch.Tensor:
    """Return ``[n_seqs, seq_len]`` (``torch.long``) windows sampled from the concatenated training stream.

    Window start offsets are drawn with a seeded CPU generator, so the result is deterministic
    for a given ``(tokenizer, dataset_name, n_seqs, seq_len, seed)``.
    ``text_loader`` overrides the dataset loader (used for offline tests).
    """
    stream = _get_token_stream(tokenizer, dataset_name, "calib", text_loader=text_loader)
    if stream.numel() <= seq_len:
        raise ValueError(f"token stream ({stream.numel()}) is not longer than seq_len={seq_len}")
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, stream.numel() - seq_len + 1, (n_seqs,), generator=g)
    return torch.stack([stream[s : s + seq_len] for s in starts.tolist()]).to(torch.long)


def get_eval_tokens(
    tokenizer,
    dataset_name: str = "wikitext2",
    seq_len: int = 2048,
    text_loader: Optional[Callable[[str, str], str]] = None,
) -> torch.Tensor:
    """Return non-overlapping ``[N_eval, seq_len]`` (``torch.long``) chunks of the held-out stream.

    The trailing partial chunk is dropped.
    """
    stream = _get_token_stream(tokenizer, dataset_name, "eval", text_loader=text_loader)
    n_chunks = stream.numel() // seq_len
    if n_chunks == 0:
        raise ValueError(f"held-out stream ({stream.numel()} tokens) is shorter than seq_len={seq_len}")
    return stream[: n_chunks * seq_len].reshape(n_chunks, seq_len).contiguous()
