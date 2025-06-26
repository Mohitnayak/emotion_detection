import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from transformers import DataCollatorWithPadding

# === Load Dataset ===
df = pd.read_csv("pipe_data/train/meld_audio_prosody.csv")
df_dev = pd.read_csv("pipe_data/dev/meld_audio_prosody.csv")

# Combine text + label
df = df[["Utterance", "Emotion"]].dropna()
df_dev = df_dev[["Utterance", "Emotion"]].dropna()

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Emotion"])
df_dev["label"] = le.transform(df_dev["Emotion"])

# Convert to HF Dataset
train_dataset = Dataset.from_pandas(df[["Utterance", "label"]])
eval_dataset = Dataset.from_pandas(df_dev[["Utterance", "label"]])

# === Load Tokenizer & Model ===
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained("kinopl/phi2-sharded-4bit")  # 4-bit tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(le.classes_),
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Apply LoRA ===
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# === Tokenization ===
def preprocess(example):
    return tokenizer(example["Utterance"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)

# === Training Args ===
training_args = TrainingArguments(
    output_dir="phi2_meld",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# === Train ===
trainer.train()

# === Save Full Model & Adapter ===
model.save_pretrained("phi2_lora_final")
tokenizer.save_pretrained("phi2_lora_final")

print("\n✅ Model saved in ./phi2_lora_final")
