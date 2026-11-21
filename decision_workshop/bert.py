"""Train a local pretrained BERT encoder to rank a task's declared actions."""

import hashlib
import json
import math
from pathlib import Path

import safetensors
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import BertModel, BertTokenizerFast


CPU_THREADS = 2


def _read_encoder(directory):
    # The Google checkpoint includes a pretraining head. The last hidden state
    # of CLS is sufficient here, so there is no pooler to initialize or train.
    encoder, info = BertModel.from_pretrained(
        directory, local_files_only=True, use_safetensors=True,
        add_pooling_layer=False, output_loading_info=True,
    )
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"incomplete pretrained BERT encoder in {directory}: {info}")
    tokenizer = BertTokenizerFast.from_pretrained(directory, local_files_only=True)
    if len(tokenizer) != encoder.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match BERT embedding table")
    return encoder, tokenizer


def _parameter_digest(module):
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        values = parameter.detach().cpu().contiguous()
        header = [name, list(values.shape), str(values.dtype)]
        digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\n")
        digest.update(values.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _option_prompts(task, tokenizer):
    question = task.get("question", task["name"].replace("_", " "))
    descriptions = task.get("descriptions", {})
    prompts = []
    for label in task["labels"]:
        name = label.replace("_", " ")
        description = descriptions.get(label)
        option = f"{name}. {description}" if description else name
        prompts.append(f"{question}\nOption: {option}")
    # Case folding and WordPiece normalization can collapse distinct label IDs.
    # A shared head cannot distinguish options with identical token sequences.
    tokens = tokenizer(prompts, add_special_tokens=False, truncation=False, verbose=False)
    if len({tuple(ids) for ids in tokens["input_ids"]}) < len(prompts):
        raise ValueError("BERT tokenizes two options identically; supply distinct option descriptions")
    return prompts


def _encode_state(tokenizer, limit, prompts, state):
    fields = []
    for key, value in sorted(state.items()):
        # Keep text intact and render numeric and boolean fields unambiguously.
        rendered = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        fields.append(f"{key.replace('_', ' ')}: {rendered}")
    state_text = "\n".join(fields)
    encoded = tokenizer([state_text] * len(prompts), prompts, padding=False,
                        truncation=False, verbose=False)
    length = max(map(len, encoded["input_ids"]))
    if length > limit:
        raise ValueError(f"BERT input requires {length} tokens, exceeding its {limit}-token limit")
    return [{name: values[i] for name, values in encoded.items()} for i in range(len(prompts))]


class _Ranker(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1, bias=False)
        nn.init.normal_(self.head.weight, std=encoder.config.initializer_range)

    def forward(self, pairs, option_count):
        hidden = self.encoder(**pairs).last_hidden_state
        return self.head(hidden[:, 0]).reshape(-1, option_count)


def _logits(model, tokenizer, examples):
    pairs = [pair for options in examples for pair in options]
    batch = tokenizer.pad(pairs, padding=True, return_tensors="pt", verbose=False)
    return model(batch, len(examples[0]))


def _train_loss(model, tokenizer, examples, targets, batch_size):
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            stop = start + batch_size
            logits = _logits(model, tokenizer, examples[start:stop])
            loss = F.cross_entropy(logits, targets[start:stop], reduction="sum")
            total += loss.item()
    return total / len(examples)


def fit_bert(task, practice, base_dir: Path, out_dir: Path, *, epochs: int, seed: int,
             batch_size: int, learning_rate: float) -> dict:
    """Fine-tune every encoder parameter and a shared scalar action-scoring head."""
    if not practice or any(row.get("split") != "train" for row in practice):
        raise ValueError("BERT training requires nonempty, train-only practice data")
    if len(task["labels"]) < 2:
        raise ValueError("BERT ranking needs at least two action labels")
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch size and finite learning rate must be positive")
    if not out_dir.is_dir() or any(out_dir.iterdir()):
        raise ValueError("BERT output must be an existing empty directory")
    if not base_dir.is_dir():
        raise ValueError("BERT base must be a local directory")
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(seed)
    encoder, tokenizer = _read_encoder(base_dir)
    model = _Ranker(encoder)
    prompts = _option_prompts(task, tokenizer)
    limit = encoder.config.max_position_embeddings
    examples = []
    for row in practice:
        try:
            examples.append(_encode_state(tokenizer, limit, prompts, row["state"]))
        except ValueError as exc:
            raise ValueError(f"training record {row['id']!r}: {exc}") from exc
    label_indices = {label: i for i, label in enumerate(task["labels"])}
    targets = torch.tensor([label_indices[row["target"]] for row in practice], dtype=torch.long)
    before = _parameter_digest(encoder)
    initial_loss = _train_loss(model, tokenizer, examples, targets, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    steps = 0
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(practice), generator=generator)
        for start in range(0, len(practice), batch_size):
            indices = order[start:start + batch_size]
            batch = [examples[i] for i in indices.tolist()]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(_logits(model, tokenizer, batch), targets[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
    final_loss = _train_loss(model, tokenizer, examples, targets, batch_size)
    after = _parameter_digest(encoder)
    if before == after or not math.isfinite(final_loss):
        raise ValueError("BERT training produced unchanged backbone weights or nonfinite loss")
    encoder.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    save_file({"weight": model.head.weight.detach().contiguous()}, out_dir / "decision_head.safetensors")
    return {
        "initial_loss": initial_loss, "final_loss": final_loss, "optimizer_steps": steps,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_parameters": sum(p.numel() for p in encoder.parameters()),
        "backbone_sha256_before": before, "backbone_sha256_after": after,
        "recipe": {
            "epochs": epochs, "seed": seed, "batch_size": batch_size,
            "learning_rate": learning_rate, "optimizer": "AdamW", "weight_decay": 0.01,
            "max_grad_norm": 1.0, "max_tokens_per_pair": limit, "device": "cpu",
            "cpu_threads": CPU_THREADS, "objective": "cross entropy over all declared options",
            "readout": "shared linear projection of final CLS state",
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
    }


def load_scorer(model_dir: Path, task):
    """Load local safetensors and return a state -> label probabilities callable."""
    if not model_dir.is_dir():
        raise ValueError("BERT checkpoint must be a local directory")
    torch.set_num_threads(CPU_THREADS)
    encoder, tokenizer = _read_encoder(model_dir)
    model = _Ranker(encoder)
    model.head.load_state_dict(load_file(model_dir / "decision_head.safetensors"))
    model.eval()
    labels = list(task["labels"])
    prompts = _option_prompts(task, tokenizer)

    @torch.inference_mode()
    def score(state):
        options = _encode_state(tokenizer, encoder.config.max_position_embeddings, prompts, state)
        logits = _logits(model, tokenizer, [options])[0]
        probs = logits.softmax(dim=0).tolist()
        if not all(map(math.isfinite, probs)):
            raise ValueError("BERT produced nonfinite option probabilities")
        return dict(zip(labels, probs))

    return score
