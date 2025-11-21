import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load Base Model
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Load YOUR Fine-Tuned Adapter
# This merges the "patch" onto the base model in memory
adapter_path = "./qwen2.5-segmentation-finetune"
model = PeftModel.from_pretrained(model, adapter_path)

# 3. Run Inference
query = "Man riding horse"
description = "The video features a person riding a horse in an open field, with a white fence in the background. The horse is trotting, and the rider is wearing a helmet. The person is also wearing a black shirt. The horse is brown in color.<|im_end|>"

prompt = f"""<|im_start|>system
You are a specialized AI for semantic segmentation. Output valid JSON only.<|im_end|>
<|im_start|>user
Focus Query: {query}
Image Description: {description}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))