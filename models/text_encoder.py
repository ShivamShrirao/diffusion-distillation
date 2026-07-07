"""Text-encoder construction + prompt-embedding utilities.

Fibo 2.0 encodes prompts with Qwen3-VL. Its pipeline normalizes chosen hidden layers and
concatenates them into bundles: the first bundle feeds the Perceiver, the rest feed the
injection blocks. ``get_qwen3vl_prompt_embeds`` returns
``(prompt_embeds, hidden_states_list, attention_mask)``: the Perceiver bundle, the list of
injection bundles, and the token mask.

`build_text_encoder` returns a ``get_prompt_embeds(prompts)`` callable that closes over
the processor/encoder and `args.max_sequence_length` for the entrypoints.
"""
from typing import List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import BriaFibo2Pipeline
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


@torch.no_grad()
def get_qwen3vl_prompt_embeds(
    processor: Qwen3VLProcessor,
    text_encoder: Qwen3VLForConditionalGeneration,
    prompts: Union[str, List[str]] = None,
    max_sequence_length: int = 2048,
    images: List[list | None] | None = None,
):
    """With ``images`` (per prompt: its images to edit, or None), each prompt is encoded with its own
    images as Fibo 2.0's pipeline encodes an edit prompt, and the batch is right-padded."""
    prompts = [prompts] if isinstance(prompts, str) else prompts
    kwargs = dict(device=text_encoder.device, max_sequence_length=max_sequence_length)
    if images is None:
        bundles, attention_mask = BriaFibo2Pipeline._get_qwen_prompt_embeds(text_encoder, processor, prompts, **kwargs)
    else:
        encoded = [
            BriaFibo2Pipeline._get_qwen_prompt_embeds(text_encoder, processor, [p], image=imgs or None, **kwargs)[0][0]
            for p, imgs in zip(prompts, images)
        ]
        lengths = torch.tensor([e.shape[1] for e in encoded], device=text_encoder.device)
        bundles = pad_sequence([e.transpose(0, 1) for e in encoded], batch_first=True).transpose(1, 2)
        attention_mask = torch.arange(bundles.shape[2], device=text_encoder.device) < lengths[:, None]
    if len(prompts) == 1:
        assert attention_mask.all()
    hidden_states = list(bundles[:, 1:].unbind(dim=1))
    return bundles[:, 0], hidden_states, attention_mask.long()


def pad_embedding(prompt_embeds: torch.Tensor, max_tokens: int):
    """Right-pad token embeddings up to `max_tokens` and return a matching mask."""
    b, seq_len, dim = prompt_embeds.shape
    padding = torch.zeros((b, max_tokens - seq_len, dim), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask = torch.zeros((b, max_tokens), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    attentions_mask[:, :seq_len] = 1
    prompt_embeds = torch.concat([prompt_embeds, padding], dim=1)
    return prompt_embeds, attentions_mask


def create_attention_matrix(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build a `(B, S, S)` attention bias (0 = keep, -inf = ignore) from a `(B, S)` mask.

    Preserves the input dtype — pass a bf16 mask to halve the bias's footprint (it is
    ~S² per sample; the 0/-inf values are exact in any float dtype)."""
    attention_matrix = torch.einsum("bi,bj->bij", attention_mask, attention_mask)
    bias = torch.full_like(attention_matrix, -torch.inf)
    return bias.masked_fill_(attention_matrix == 1, 0.0)


def freeze_model(model, device):
    """Put a frozen model on the compute device in eval mode."""
    model = model.to(device)
    return model.eval().requires_grad_(False)


def init_text_encoder(model_path: str, device, weight_dtype):
    """Load the Qwen3-VL text encoder + processor and return its prompt-embed callable."""
    print("Loading Qwen3-VL text encoder")
    processor = Qwen3VLProcessor.from_pretrained(model_path, subfolder="processor")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=weight_dtype, low_cpu_mem_usage=True
    )
    text_encoder = freeze_model(text_encoder, device)
    return processor, text_encoder, get_qwen3vl_prompt_embeds


def build_text_encoder(args, device, weight_dtype):
    """Returns `(processor, encoder, get_prompt_embeds_fn)` for use in the training loop.

    The returned callable has signature `(prompts, images=None) -> (embeds, hidden_states, attn_mask)`
    and closes over `args.max_sequence_length`.
    """
    processor, text_encoder, encoder_callable = init_text_encoder(
        args.pretrained_model_name_or_path, device, weight_dtype
    )

    def get_prompt_embeds(prompts, images=None):
        return encoder_callable(
            processor,
            text_encoder,
            prompts=prompts,
            max_sequence_length=args.max_sequence_length,
            images=images,
        )

    return processor, text_encoder, get_prompt_embeds
