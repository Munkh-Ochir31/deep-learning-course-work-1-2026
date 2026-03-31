"""
Stage 1: Continued Pretraining - Монгол хэлний текст дээр дахин сургах

Зорилго: Qwen2.5-3B загвар монгол хэлний текстийг илүү сайн ойлгох болгох.

Хэрхэн ажилладаг:
- Causal Language Modeling (CLM): Загварт өмнөх token-уудыг өгөөд дараагийн token-г таахыг сургана.
  Жишээ: "Монгол улсын нийслэл" -> "Улаанбаатар"
- LoRA: Бүх параметрийг сургахын оронд жижиг adapter (rank=16) нэмж,
  зөвхөн түүнийг сургана. Ингэснээр 8GB VRAM-д багтана.
- 4-bit quantization: Загварыг 4 бит болгож шахна. Санах ой хэмнэнэ.
"""

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.integrations import TensorBoardCallback
import json
import math
import os
from datetime import datetime

# ============================================================
# 1. Тохиргоо (Config)
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B"  # Суурь загвар
DATA_PATH = "data/OneDrive_1_3-29-2026/mn_61785_test"  # Монгол текст dataset
OUTPUT_DIR = "models/stage1_pretrained"  # Сургасан загварыг хадгалах зам
LOG_DIR = "logs/stage1"                  # Log файлуудын зам

# Сургалтын hyperparameter-ууд
NUM_EPOCHS = 1          # Dataset-ийг хэдэн удаа давтах (1 нь хангалттай, их бол overfit болно)
BATCH_SIZE = 2           # Нэг удаад хэдэн жишээ боловсруулах (VRAM-аас хамаарна)
GRADIENT_ACCUMULATION = 8  # 8 batch хуримтлуулаад 1 удаа weight шинэчлэх
                           # Effective batch size = 2 * 8 = 16
LEARNING_RATE = 2e-4     # Сургалтын хурд (хэт их = тогтворгүй, хэт бага = удаан)
MAX_SEQ_LENGTH = 512     # Текстийн дээд урт (token-оор). Урт текстийг таслана.

# ============================================================
# 2. Загвар ачаалах (Model Loading)
# ============================================================

print("=" * 50)
print("Stage 1: Continued Pretraining")
print("=" * 50)

# 4-bit quantization тохиргоо
# Загварын жинг 32-bit float -> 4-bit integer болгож шахна
# Ингэснээр ~12GB загвар -> ~3GB болно
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit горимоор ачаалах
    bnb_4bit_quant_type="nf4",            # NormalFloat4 - хамгийн сайн шахалтын төрөл
    bnb_4bit_compute_dtype=torch.float16, # Тооцооллыг float16-аар хийнэ
    bnb_4bit_use_double_quant=True,       # Давхар шахалт (нэмэлт санах ой хэмнэнэ)
)

print(f"\n[1/5] Загвар ачаалж байна: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",            # GPU автоматаар сонгоно
    torch_dtype=torch.float16,
)

# Tokenizer-д pad token тохируулах (байхгүй бол алдаа гарна)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ============================================================
# 3. LoRA тохиргоо (Parameter-Efficient Fine-Tuning)
# ============================================================

# LoRA (Low-Rank Adaptation):
# - Бүх параметрийг сургахын оронд жижиг матриц (adapter) нэмнэ
# - r=16: adapter-ийн хэмжээ (их = илүү сурах чадвар, бас илүү санах ой)
# - target_modules: аль давхаргуудыг сургах вэ (attention болон MLP давхаргууд)
# - lora_alpha: LoRA-ийн масштаб (ихэвчлэн r-ийн 2 дахин)
# - lora_dropout: Overfitting-ээс сэргийлэх (10%)

print("\n[2/5] LoRA тохируулж байна...")
model = prepare_model_for_kbit_training(model)  # 4-bit загварыг сургалтад бэлдэх

lora_config = LoraConfig(
    r=16,                                          # Rank - adapter-ийн хэмжээ
    lora_alpha=32,                                 # Масштаб (alpha/r = scaling factor)
    target_modules=["q_proj", "k_proj", "v_proj",  # Attention давхаргууд
                     "o_proj", "gate_proj",         # MLP давхаргууд
                     "up_proj", "down_proj"],
    lora_dropout=0.1,                              # 10% dropout
    bias="none",                                   # Bias сургахгүй
    task_type="CAUSAL_LM",                         # Causal Language Modeling даалгавар
)

model = get_peft_model(model, lora_config)

# Сургах параметрийн тоог харуулах
model.print_trainable_parameters()
# Жишээ: "trainable params: 27M || all params: 3B || trainable%: 0.9%"
# Зөвхөн 0.9%-ийг нь сургаж байна!

# ============================================================
# 4. Dataset бэлдэх
# ============================================================

print(f"\n[3/5] Dataset ачаалж байна: {DATA_PATH}")
dataset = load_from_disk(DATA_PATH)
print(f"  Нийт жишээ: {len(dataset)}")

# Текстийг token болгох функц
# Загвар текстийг шууд ойлгодоггүй - эхлээд тоон token болгох хэрэгтэй
# Жишээ: "Сайн байна уу" -> [12045, 8834, 2231, 445]
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,          # MAX_SEQ_LENGTH-аас урт бол таслах
        max_length=MAX_SEQ_LENGTH,
        padding=False,            # Padding хийхгүй (DataCollator хийнэ)
    )

print("  Tokenizing...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,                 # Batch-ээр боловсруулах (хурдан)
    remove_columns=["text"],      # Анхны текст баганыг устгах (token-ууд л хэрэгтэй)
    num_proc=4,                   # 4 CPU цөм ашиглах (параллел)
    desc="Tokenizing",
)

# Train/validation хуваах (90% сургалт, 10% шалгах)
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
# Eval dataset-ийг хязгаарлах (GPU timeout-аас сэргийлэх)
MAX_EVAL_SAMPLES = 500
if len(eval_dataset) > MAX_EVAL_SAMPLES:
    eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# DataCollator: batch үүсгэх, padding нэмэх, label үүсгэх
# CLM-д label = input_ids (дараагийн token-г таах)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Masked LM биш, Causal LM (зүүнээс баруун руу)
)

# ============================================================
# 5. Logging тохиргоо
# ============================================================

# TrainerCallback: Сургалтын явцад автоматаар дуудагддаг функцүүд
# on_log()   - алхам бүрийн loss, lr зэргийг бичнэ
# on_evaluate() - eval хийх бүрд perplexity тооцоолно
# on_train_end() - сургалт дууссаны дараа нийт үр дүнг хадгална
from transformers import TrainerCallback

class LogCallback(TrainerCallback):
    """Сургалтын явцыг JSON файлд бичих callback"""

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.json")
        self.train_losses = []    # Loss-ийн түүх (график зурахад)
        self.eval_results = []    # Eval үр дүнгүүд
        self.start_time = None
        print(f"  Log файл: {self.log_file}")

    def on_train_begin(self, args, state, control, **kwargs):
        """Сургалт эхлэхэд"""
        self.start_time = datetime.now()
        self._write_log({
            "event": "train_start",
            "time": self.start_time.isoformat(),
            "config": {
                "model": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRADIENT_ACCUMULATION,
                "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION,
                "lr": LEARNING_RATE,
                "max_seq_length": MAX_SEQ_LENGTH,
                "lora_r": 16,
                "lora_alpha": 32,
            }
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Алхам бүрд loss, learning rate бичих"""
        if logs is None:
            return
        entry = {
            "event": "step",
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
        }
        if "loss" in logs:
            entry["train_loss"] = round(logs["loss"], 4)
            self.train_losses.append(logs["loss"])
        if "learning_rate" in logs:
            entry["lr"] = logs["learning_rate"]
        if "eval_loss" in logs:
            entry["eval_loss"] = round(logs["eval_loss"], 4)
            entry["perplexity"] = round(math.exp(logs["eval_loss"]), 2)
            self.eval_results.append(entry.copy())
        # VRAM хэрэглээ
        if torch.cuda.is_available():
            entry["vram_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)

        self._write_log(entry)

    def on_train_end(self, args, state, control, **kwargs):
        """Сургалт дуусахад нийт үр дүнг хадгалах"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        summary = {
            "model": MODEL_NAME,
            "total_steps": state.global_step,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": round(duration.total_seconds() / 60, 1),
            "final_train_loss": round(self.train_losses[-1], 4) if self.train_losses else None,
            "best_eval_loss": round(min(e["eval_loss"] for e in self.eval_results), 4) if self.eval_results else None,
            "best_perplexity": round(min(e["perplexity"] for e in self.eval_results), 2) if self.eval_results else None,
            "loss_history_length": len(self.train_losses),
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n  Summary хадгалагдсан: {self.summary_file}")

    def _write_log(self, entry):
        """Нэг мөр JSONL бичих"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

log_callback = LogCallback(LOG_DIR)

# ============================================================
# 6. Сургалт (Training)
# ============================================================

print("\n[4/5] Сургалт эхэлж байна...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,                    # Regularization (overfitting-ээс сэргийлэх)
    warmup_ratio=0.03,                    # Эхний 3% алхамд LR аажмаар нэмэгдэнэ
    lr_scheduler_type="cosine",           # LR хуваарь: cosine (аажмаар буурна)
    logging_steps=10,                     # 10 алхам тутамд loss хэвлэх
    logging_dir=LOG_DIR,                  # TensorBoard log зам
    save_steps=500,                       # 500 алхам тутамд checkpoint хадгалах
    eval_strategy="steps",
    eval_steps=500,                       # 500 алхам тутамд eval хийх
    save_total_limit=2,                   # Хамгийн сүүлийн 2 checkpoint л хадгалах
    fp16=True,                            # Float16 ашиглах (хурдан + санах ой хэмнэх)
    report_to="none",                     # wandb г.м. ашиглахгүй
    dataloader_num_workers=2,
    gradient_checkpointing=True,          # VRAM хэмнэх (тооцооллыг дахин хийнэ)
    optim="paged_adamw_8bit",             # 8-bit optimizer (VRAM хэмнэх)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[log_callback],  # Log callback залгах
)

# Сургалт эхлүүлэх (checkpoint байвал үргэлжлүүлнэ)
resume_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_checkpoint = os.path.join(OUTPUT_DIR, latest)
        print(f"  Checkpoint олдлоо: {resume_checkpoint}")

train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

# ============================================================
# 7. Хадгалах ба Үнэлэх
# ============================================================

print("\n[5/5] Загвар хадгалж байна...")

# LoRA adapter-ийг хадгалах
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Сургалтын үр дүнг хэвлэх
print("\n" + "=" * 50)
print("Сургалтын Үр дүн:")
print(f"  Train Loss: {train_result.metrics['train_loss']:.4f}")

# Eval дээр perplexity тооцоолох
# Perplexity = e^(loss) - загвар текстийг хэр сайн таах вэ
# Бага perplexity = сайн (загвар текстийг сайн ойлгож байна)
eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"  Perplexity: {perplexity:.2f}")
print(f"\n  Загвар хадгалагдсан: {OUTPUT_DIR}")
print("=" * 50)
