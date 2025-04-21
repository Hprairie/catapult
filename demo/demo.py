from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import time
import copy
import catapult
from transformers.models.llama.modeling_llama import LlamaAttention
import torch.nn as nn

@catapult.jit(
    kernel_path="attention_tk.cuh",
    kernel_name="attend_ker",
    kernel_param="globals",
    template_kernel=["D"],
    template_params=["D"],
)
def attend(Qg, Kg, Vg):
    Og = torch.empty_like(Qg)
    attend.kernel(Qg, Kg, Vg, Og, D=64)
    return Og

class CatapultLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype, device=self.q_proj.weight.device)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        out = attend(q, k, v)

        out = self.o_proj(out)

        return out, None, None

import re

def replace_llama_attention(module: nn.Module, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, LlamaAttention):
            m = re.search(r"layers\.(\d+)", full_name)
            if m is None:
                raise ValueError(f"Could not extract layer index from module path: {full_name}")
            layer_idx = int(m.group(1))

            catapult_attn = CatapultLlamaAttention(child.config, layer_idx=layer_idx)
            setattr(module, name, catapult_attn)
        else:
            replace_llama_attention(child, full_name)


# model load

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_ref = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
model_ref.eval()

model_catapult = copy.deepcopy(model_ref)
replace_llama_attention(model_catapult)
model_catapult.eval()

# prompt

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

max_new_tokens = 50

# gen

def generate(model, input_ids, max_new_tokens=50):
    generated = input_ids.clone()
    timings = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            out = model(input_ids=generated)
            logits = out.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
    return generated, timings


# stats

print("Generating with reference model...")
tokens_ref, timings_ref = generate(model_ref, input_ids, max_new_tokens)

print("Generating with catapult model...")
tokens_catapult, timings_catapult = generate(model_catapult, input_ids, max_new_tokens)

decoded_ref = tokenizer.decode(tokens_ref[0], skip_special_tokens=True)
decoded_catapult = tokenizer.decode(tokens_catapult[0], skip_special_tokens=True)

print("\n--- Reference Output ---")
print(decoded_ref)
print("\n--- Catapult Output ---")
print(decoded_catapult)

def show_timings(label, times):
    avg_ms = sum(times) / len(times)
    print(f"{label} — Avg per token: {avg_ms:.2f} ms, Throughput: {1000.0 / avg_ms:.2f} tok/sec")

show_timings("Reference", timings_ref)
show_timings("Catapult", timings_catapult)

plt.plot(timings_ref, label="Reference")
plt.plot(timings_catapult, label="Catapult")
plt.xlabel("Token Index")
plt.ylabel("Time per token (ms)")
plt.legend()
plt.title("Token Generation Time Comparison")
plt.grid(True)
plt.show()
