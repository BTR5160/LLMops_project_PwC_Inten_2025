import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import torch

# ✅ Configurations
MODEL_PATH = "./gemma-3-1b-it"  # dossier du modèle local
DATA_PATH = "data/output/fine_tune.json"
OUTPUT_DIR = "models/gemma-finetuned"
USE_WANDB = os.environ.get("USE_WANDB", "0") == "1"

if USE_WANDB:
    import wandb
    wandb.init(project="llmops-finetune", name="gemma-local-qLoRA")

# ✅ Charger les données JSONL
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f.readlines()]

dataset = Dataset.from_list(raw_data)

# ✅ Tokenizer + Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
    trust_remote_code=True
)

# ✅ Préparation au fine-tuning avec LoRA
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adapte selon le modèle
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, peft_config)

# ✅ Préparation des données tokenisées
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Arguments d'entraînement
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="wandb" if USE_WANDB else "none",
    run_name="gemma-finetune" if USE_WANDB else None,
    logging_dir=f"{OUTPUT_DIR}/logs",
)

# ✅ Lancement de l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# ✅ Sauvegarde
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if USE_WANDB:
    wandb.finish()

print("✅ Fine-tuning terminé. Modèle enregistré dans :", OUTPUT_DIR)
