import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==============================
# 1. CONFIGURATION
# ==============================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # The 3B model (~6.2GB)
OUTPUT_DIR = "./qwen2.5-segmentation-finetune"
DATA_FILE = "/home/harsh/AI/Sa2VA/Sa2VA/finetune_qwen/training_data/train_257.json"  # <--- Ensure your JSON file is named this

# ==============================
# 2. DATA PREPARATION
# ==============================
def format_data_from_json(json_path):
    """
    Reads the specific dictionary format {"1": {...}, "2": {...}} 
    and converts it into ChatML style messages.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}. Please make sure your data file exists.")
        return []

    formatted_samples = []

    # Iterate through keys "1", "2", "3"...
    for key, item in raw_data.items():
        # 1. Extract Inputs
        query = item.get("FOCUS_QUERY", "")
        desc = item.get("IMAGE_DESCRIPTION", "")
        
        # 2. Extract Outputs (The Sentences)
        # Filter for keys starting with "sentence" (sentence1, sentence2...)
        output_dict = {k: v for k, v in item.items() if k.startswith("sentence")}
        output_json_str = json.dumps(output_dict, ensure_ascii=False)

        # 3. Construct the Prompt (ChatML format)
        messages = [
            {
                "role": "system", 
                "content": "You are a specialized AI for semantic segmentation. You extract segmentation targets from a query and visual description. Output valid JSON only."
            },
            {
                "role": "user", 
                "content": f"Focus Query: {query}\nImage Description: {desc}"
            },
            {
                "role": "assistant", 
                "content": output_json_str
            }
        ]
        
        formatted_samples.append({"messages": messages})
    
    return formatted_samples

print(f"Loading and formatting data from {DATA_FILE}...")
data = format_data_from_json(DATA_FILE)

if not data:
    raise ValueError("No data found! Check your json file path.")

dataset = Dataset.from_list(data)
print(f"Loaded {len(dataset)} samples.")

# ==============================
# 3. LOAD MODEL & TOKENIZER
# ==============================
print("Loading model in 4-bit quantization...")

# 4-bit Config to fit 8GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Fix padding issues

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for training (Gradient Checkpointing reduces VRAM usage significantly)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Must be False for training

# ==============================
# 4. LoRA CONFIGURATION
# ==============================
peft_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=16,           # Alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen target modules (Linear layers)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==============================
# 5. TRAINING ARGUMENTS
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # Keep 1 for 8GB VRAM
    gradient_accumulation_steps=4,  # Simulates batch size of 4
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=5,             # Increased to 5 because dataset is small (45 items)
    save_steps=50,
    optim="paged_adamw_32bit",      # Saves VRAM
    fp16=True,                      # Mixed precision
    warmup_ratio=0.1,
    report_to="none"                # No wandb/tensorboard
)

# ==============================
# 6. START TRAINING
# ==============================
print("Starting training...")

# Helper function to apply chat template during training
# This Fixes the "ValueError: passed packing=False" error
def formatting_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return texts

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func, # Applies the chat template
    max_seq_length=1024,             # Fixes the missing argument warning
)

trainer.train()

# ==============================
# 7. SAVE ADAPTER
# ==============================
print(f"Saving adapter model to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning complete! You can now run inference.py")