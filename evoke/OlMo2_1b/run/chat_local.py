"""
Chat with the local OlMo2-1B-Instruct model.
Conversation history is kept in RAM.

Run from MINDOLOGY root:
    python evoke/OlMo2_1b/run/chat_local.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
THIS_DIR = PROJECT_ROOT / "evoke" / "OlMo2_1b" / "run"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

import torch

from loader import build_model_and_tokenizer


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens=500,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=None,
    pad_token_id=None,
):
    """
    Generate tokens with KV-cache support.

    input_ids:     (batch, seq_len)
    attention_mask:(batch, seq_len)
    """
    device = input_ids.device
    past_key_values = None
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        if past_key_values is None:
            # first forward: process full prompt
            logits, _, past_key_values = model(
                input_ids=generated,
                attention_mask=attention_mask,
            )
        else:
            # subsequent forwards: only the last token
            logits, _, past_key_values = model(
                input_ids=generated[:, -1:],
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

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def main():
    print("Loading OlMo2-1B-Instruct...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = build_model_and_tokenizer(device=device)
    print(f"Model loaded on {device}\n")

    messages = []
    print("OlMo2-1B chat. Type 'quit', 'exit', or 'q' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            break

        messages.append({"role": "user", "content": user_text})

        # build prompt with chat template
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        prompt_tensor = encoded.input_ids.to(device)

        # attention mask: all real tokens
        attention_mask = encoded.attention_mask.to(device)

        print("Generating...", end=" ", flush=True)
        outputs = generate(
            model,
            prompt_tensor,
            attention_mask,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        print("done")

        # decode only the new tokens
        reply_ids = outputs[0][prompt_tensor.shape[-1]:]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True)
        print(f"Bot: {reply}\n")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
