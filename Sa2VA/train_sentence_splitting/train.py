from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# --- Configuration ---
max_seq_length = 2048 # Supports long contexts
dtype = None # None = auto detection (Float16 for you)
load_in_4bit = True # CRITICAL for your 8GB GPU

# 1. Load the Model (Qwen 2.5 3B)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Add LoRA Adapters (This makes training efficient)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # 0 is optimized
    bias = "none",
    use_gradient_checkpointing = True, # Saves VRAM
    random_state = 3407,
)

# 3. Load Your Dataset
# Ensure your dataset.jsonl is formatted as shown in the previous step
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# 4. Define Training Arguments (Optimized for 8GB VRAM)
training_args = TrainingArguments(
    per_device_train_batch_size = 2, # Low batch size to save VRAM
    gradient_accumulation_steps = 4, # Simulates a batch size of 8
    warmup_steps = 5,
    max_steps = 60, # For 50-100 examples, 60-100 steps is usually enough
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit", # 8-bit optimizer saves memory
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
)

# 5. Train
print("--- Starting Training ---")
trainer.train()

# 6. Save the Model
# This saves the 'adapters' (the learned part)
model.save_pretrained("my_finetuned_qwen")
tokenizer.save_pretrained("my_finetuned_qwen")

# 7. Merge to GGUF (Optional, for Ollama)
# model.save_pretrained_merged("merged_model", tokenizer, save_method="gguf_q4_k_m")
print("--- Done! Adapters saved to 'my_finetuned_qwen' ---")