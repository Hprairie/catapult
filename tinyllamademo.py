from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
model.eval()

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

max_new_tokens = 200
generated = input_ids
timings = []

print("Generating:")
print("Prompt:", tokenizer.convert_ids_to_tokens(input_ids[0]))

with torch.no_grad():
    for _ in range(max_new_tokens):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        outputs = model(input_ids=generated, attention_mask=None)
        logits = outputs.logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        end_event.record()

        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

        token_str = tokenizer.convert_ids_to_tokens(next_token[0])[0]
        print(token_str, end=" ", flush=True)

print("\n\n--- Final Output ---")
decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
print(decoded.strip())

avg_ms = sum(timings) / len(timings)
print("\n--- Inference Performance Summary ---")
print(f"Average time per token: {avg_ms:.2f} ms")
print(f"Throughput: {1000.0 / avg_ms:.2f} tokens/sec")

plt.plot(timings)
plt.xlabel("Token Index")
plt.ylabel("Time per token (ms)")
plt.title("Inference Token Timing")
plt.grid(True)
plt.show()

