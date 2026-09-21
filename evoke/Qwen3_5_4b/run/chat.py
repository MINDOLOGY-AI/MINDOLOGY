"""
Chat with the local Qwen3.5-4B model (text branch).
Conversation history is kept in RAM.

Run from MINDOLOGY root as a module:
    python -m mind.evoke.Qwen3_5_4b.run.chat
"""

import logging

import torch
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from evoke.Qwen3_5_4b.run.asy_prompt import asy_prompt
from evoke.Qwen3_5_4b.run.config import QWEN3_5_4B_CONFIG
from evoke.Qwen3_5_4b.run.loader import build_model_and_tokenizer

# use SDPA (flash kernels) for the full-attention layers in chat; the config
# default stays eager so interp code is unaffected
QWEN3_5_4B_CONFIG.use_flash_attention = True

# bitsandbytes spams "MatMul8bitLt: inputs will be cast..." once per int8 matmul; harmless
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

# effectively uncapped for chat; generation still stops at EOS (<|im_end|>)
MAX_NEW_TOKENS = 8192

# Qwen3.5 thinks by default; the chat template can disable it
ENABLE_THINKING = False

# persistent system message; set to None to disable
SYSTEM_PROMPT = asy_prompt

console = Console()


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_ids=None,
    past_key_values=None,
    on_token=None,
):
    """
    Generate tokens with the hybrid (KV + recurrent) cache.

    input_ids:     (batch, seq_len) -- only the NEW tokens to process; anything
                   before them must already live in past_key_values
    attention_mask:(batch, cache_len + seq_len)
    past_key_values: optional incoming cache from previous turns
    on_token:      optional callback called with the full generated tensor
                   after each new token (used for streaming output)
    returns:       (generated, past_key_values)
    """
    device = input_ids.device
    generated = input_ids.clone()
    cur = input_ids

    for _ in range(max_new_tokens):
        # first forward processes the whole new chunk; later ones just the
        # last sampled token
        logits, _, past_key_values = model(
            input_ids=cur,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        next_token_logits = logits[:, -1, :]

        if do_sample:
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature

            # top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=-1,
        )
        cur = next_token

        if on_token is not None:
            on_token(generated)

        if eos_token_ids is not None and next_token.item() in eos_token_ids:
            break

    return generated, past_key_values


def main():
    print("Loading Qwen3.5-4B...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = build_model_and_tokenizer(device=device)
    print(f"Model loaded on {device}\n")

    # stop on <|im_end|> (config eos) or the tokenizer's own eos
    eos_token_ids = {config.eos_token_id, tokenizer.eos_token_id}
    eos_token_ids.discard(None)

    print("Qwen3.5-4B chat. Type 'quit', 'exit', or 'q' to exit.\n")

    # generation prompt, mirroring the chat template
    GEN_PROMPT = "<|im_start|>assistant\n" + (
        "<think>\n" if ENABLE_THINKING else "<think>\n\n</think>\n\n"
    )

    # the cache persists across turns; history is NEVER re-rendered or re-fed.
    # turn 0 goes through the chat template (system prompt, tools, etc.);
    # later turns hand-build the small delta: "\n<|im_start|>user\n...".
    # after every reply we feed the closing <|im_end|>, so the cache always
    # ends exactly where the template's assistant turn would.
    past = None
    n_cached = 0  # tokens the model has ingested so far
    first_turn = True

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            break

        if first_turn:
            messages = []
            if SYSTEM_PROMPT is not None:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": user_text})
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
            delta = encoded["input_ids"]
            if isinstance(delta[0], list):
                delta = delta[0]  # unbatch: single conversation
            first_turn = False
        else:
            delta_text = (
                "\n<|im_start|>user\n" + user_text + "<|im_end|>\n" + GEN_PROMPT
            )
            delta = tokenizer(delta_text, add_special_tokens=False)["input_ids"]

        delta_tensor = torch.tensor([delta], device=device)
        attention_mask = torch.ones(
            (1, n_cached + len(delta)), dtype=torch.long, device=device
        )

        # stream: re-decode the whole reply each step and re-render it as
        # markdown in place (re-decoding avoids mangling multi-token chars)
        prompt_len = delta_tensor.shape[-1]

        def stream(generated):
            text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
            live.update(Markdown(text))

        console.print("Bot: ", end="")
        with Live(Markdown(""), console=console, refresh_per_second=12) as live:
            outputs, past = generate(
                model,
                delta_tensor,
                attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_ids=eos_token_ids,
                past_key_values=past,
                on_token=stream,
            )
        console.print()

        # close the turn: the final sampled token was never fed (generation
        # stops when it is produced), and if the model hit the cap without
        # emitting <|im_end|> we inject one -- either way the cache ends right
        # after the assistant turn's <|im_end|>
        closing = outputs[0][-1:].tolist()
        if closing[0] not in eos_token_ids:
            closing.append(config.eos_token_id)
        n_cached = past.get_seq_length()
        _, _, past = model(
            input_ids=torch.tensor([closing], device=device),
            attention_mask=torch.ones(
                (1, n_cached + len(closing)), dtype=torch.long, device=device
            ),
            past_key_values=past,
        )
        n_cached = past.get_seq_length()


if __name__ == "__main__":
    main()
