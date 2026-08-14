"""Projector and continuous GraphToken injection for Stage 2 QA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .stage2_data import GRAPH_SLOT


class GraphProjector(nn.Module):
    def __init__(
        self,
        graph_dim: int = 128,
        hidden_dim: int = 512,
        llm_dim: int = 4096,
        num_tokens: int = 4,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.hidden_dim = int(hidden_dim)
        self.llm_dim = int(llm_dim)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.Linear(self.graph_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_tokens * self.llm_dim),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=float(initializer_range))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, topology_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.net(topology_embedding.float())
        return projected.view(-1, self.num_tokens, self.llm_dim)


@dataclass
class InjectedBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor | None
    prompt_lengths: torch.Tensor
    graph_token_rms: float
    text_token_rms: float


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def render_qwen_prompt(tokenizer, question: str) -> str:
    user_text = (
        "The following continuous vectors represent the topology of one rooted graph.\n"
        f"{GRAPH_SLOT}\n"
        "Answer the question using only those vectors. Return one integer and no other text.\n"
        f"Question: {question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _find_unique_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> tuple[int, int]:
    if not subsequence:
        raise ValueError("Graph slot tokenization is empty")
    starts = [
        start
        for start in range(len(sequence) - len(subsequence) + 1)
        if list(sequence[start : start + len(subsequence)]) == list(subsequence)
    ]
    if len(starts) != 1:
        raise ValueError(f"Graph slot must tokenize exactly once, found {len(starts)} matches")
    return starts[0], starts[0] + len(subsequence)


def prompt_token_ids(tokenizer, question: str, max_length: int) -> tuple[list[int], int, int]:
    prompt = render_qwen_prompt(tokenizer, question)
    if prompt.count(GRAPH_SLOT) != 1:
        raise ValueError("Rendered prompt must contain the Graph slot exactly once")
    # Fast tokenizers expose character offsets, which remains correct even if
    # a BPE token joins a sentinel boundary with an adjacent newline.
    try:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError):
        encoded = tokenizer(prompt, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is not None:
        character_start = prompt.index(GRAPH_SLOT)
        character_end = character_start + len(GRAPH_SLOT)
        overlapping = [
            index for index, (start, end) in enumerate(offsets)
            if end > character_start and start < character_end
        ]
        if not overlapping or overlapping != list(range(overlapping[0], overlapping[-1] + 1)):
            raise ValueError("Graph slot offsets are missing or non-contiguous")
        start, end = overlapping[0], overlapping[-1] + 1
    else:
        slot_ids = tokenizer(GRAPH_SLOT, add_special_tokens=False)["input_ids"]
        start, end = _find_unique_subsequence(ids, slot_ids)
    if len(ids) - (end - start) > int(max_length):
        raise ValueError(f"Prompt exceeds max_input_length={max_length}")
    return list(ids), start, end


def _answer_ids(tokenizer, answer: int) -> list[int]:
    ids = list(tokenizer(str(int(answer)), add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define eos_token_id")
    return ids + [int(eos)]


def build_injected_batch(
    *,
    tokenizer,
    embedding_layer: nn.Module,
    questions: Sequence[str],
    graph_tokens: torch.Tensor | None,
    answers: Sequence[int] | None,
    max_length: int,
) -> InjectedBatch:
    """Build a padded batch; only answer/EOS positions receive labels."""
    if graph_tokens is not None and len(questions) != graph_tokens.shape[0]:
        raise ValueError("Question and GraphToken batch sizes differ")
    if answers is not None and len(questions) != len(answers):
        raise ValueError("Question and answer batch sizes differ")
    device = graph_tokens.device if graph_tokens is not None else embedding_layer.weight.device
    dtype = embedding_layer.weight.dtype
    rows: list[torch.Tensor] = []
    row_labels: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    text_squares: list[torch.Tensor] = []
    graph_squares: list[torch.Tensor] = []
    for index, question in enumerate(questions):
        ids, slot_start, slot_end = prompt_token_ids(tokenizer, question, max_length)
        before_ids, after_ids = ids[:slot_start], ids[slot_end:]
        before = embedding_layer(torch.tensor(before_ids, dtype=torch.long, device=device))
        after = embedding_layer(torch.tensor(after_ids, dtype=torch.long, device=device))
        parts = [before]
        if graph_tokens is not None:
            token_row = graph_tokens[index].to(dtype=dtype)
            parts.append(token_row)
            graph_squares.append(token_row.float().square().mean())
        parts.append(after)
        prompt_row = torch.cat(parts, dim=0)
        if prompt_row.shape[0] > int(max_length):
            raise ValueError(
                f"Injected prompt length {prompt_row.shape[0]} exceeds max_input_length={max_length}"
            )
        text_squares.extend([before.float().square().mean(), after.float().square().mean()])
        prompt_lengths.append(prompt_row.shape[0])
        if answers is None:
            rows.append(prompt_row)
            continue
        target_ids = _answer_ids(tokenizer, int(answers[index]))
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
        target_embeds = embedding_layer(target_tensor)
        rows.append(torch.cat([prompt_row, target_embeds], dim=0))
        row_labels.append(
            torch.cat(
                [
                    torch.full((prompt_row.shape[0],), -100, dtype=torch.long, device=device),
                    target_tensor,
                ]
            )
        )

    max_row = max(row.shape[0] for row in rows)
    hidden = rows[0].shape[-1]
    inputs = torch.zeros((len(rows), max_row, hidden), dtype=dtype, device=device)
    mask = torch.zeros((len(rows), max_row), dtype=torch.long, device=device)
    labels = (
        torch.full((len(rows), max_row), -100, dtype=torch.long, device=device)
        if answers is not None
        else None
    )
    for index, row in enumerate(rows):
        length = row.shape[0]
        inputs[index, :length] = row
        mask[index, :length] = 1
        if labels is not None:
            labels[index, :length] = row_labels[index]
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask == 0, 0)
    graph_rms = float(torch.stack(graph_squares).mean().sqrt()) if graph_squares else 0.0
    text_rms = float(torch.stack(text_squares).mean().sqrt())
    return InjectedBatch(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=positions,
        labels=labels,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        graph_token_rms=graph_rms,
        text_token_rms=text_rms,
    )


@torch.no_grad()
def greedy_generate_from_embeds(
    model: nn.Module,
    batch: InjectedBatch,
    *,
    eos_token_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """Deterministic batched decoding that supports continuous input embeddings."""
    outputs = model(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=True,
        return_dict=True,
    )
    batch_indices = torch.arange(batch.inputs_embeds.shape[0], device=batch.inputs_embeds.device)
    last_positions = batch.attention_mask.sum(dim=1) - 1
    next_token = outputs.logits[batch_indices, last_positions].argmax(dim=-1).to(batch.inputs_embeds.device)
    generated = [next_token]
    finished = next_token.eq(int(eos_token_id))
    past = outputs.past_key_values
    attention_mask = batch.attention_mask
    for _ in range(1, int(max_new_tokens)):
        if bool(finished.all()):
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        # Right-padded prompts have different logical lengths. Preserve each
        # sample's RoPE position rather than using the padded cache length.
        step_position_ids = attention_mask.sum(dim=-1, keepdim=True) - 1
        step = model(
            input_ids=next_token[:, None],
            attention_mask=attention_mask,
            position_ids=step_position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        next_token = step.logits[:, -1].argmax(dim=-1).to(batch.inputs_embeds.device)
        next_token = torch.where(finished, torch.full_like(next_token, int(eos_token_id)), next_token)
        generated.append(next_token)
        finished |= next_token.eq(int(eos_token_id))
        past = step.past_key_values
    return torch.stack(generated, dim=1)
