"""
Stage 1 Model Inspection Script - Моделийн үзүүлэлтүүдийг шалгах
Сургасан моделын параметрүүд, цаг хугацаа, үзүүлэлтүүдийг харуулна.
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import sys

# ============================================================
# Тохиргоо (Configuration)
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B"
STAGE1_MODEL_PATH = "models/stage1_pretrained"
OUTPUT_REPORT = "src/MODEL_INSPECTION_REPORT.md"

# Төрвөлийн сорилтуудын жишээнүүд
MONGOLIAN_PROMPTS = [
    "Монгол улсын нийслэл нь",
    "Хиймэл оюун ухаан гэж",
    "Монгол хэл дээр сайн байна гэж үнэ",
    "Их сургуулийн боловсрол",
    "Компьютер шинжлэл нь",
]

# ============================================================
# UtilityFunctions
# ============================================================

def format_size(size_bytes):
    """Байтыг хүний уншиж болох хэлбэрт хөрвүүлэх"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def count_parameters(model):
    """Моделийн нийт параметрүүдийн тоог тоолох"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_memory_usage():
    """VRAM-ны хэрэглээг авах"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    return 0, 0

def load_training_logs():
    """Сургалтын логуудыг уншиж авах"""
    log_path = "logs/stage1/training_log.jsonl"
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                logs.append(json.loads(line))
    return logs

def load_training_summary():
    """Сургалтын хураангуй мэдээлэлийг авах"""
    summary_path = "logs/stage1/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None

def load_adapter_config():
    """LoRA adapter тохиргоог уншиж авах"""
    config_path = "models/stage1_pretrained/adapter_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# ============================================================
# Үндсэн үйлдлүүд
# ============================================================

def main():
    print("="*70)
    print("STAGE 1 MODEL INSPECTION - Моделийн үзүүлэлтүүдийг шалгаж байна")
    print("="*70)
    
    report_lines = []
    report_lines.append("# Stage 1 Model Inspection Report\n")
    report_lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # ============================================================
    # 1. Моделийн архитектур болон параметрүүдийн тоо
    # ============================================================
    print("\n[1/6] Нүүр хэсгийн загварыг ачаалж байна...")
    report_lines.append("## 1. Base Model Information\n")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        base_total, base_trainable = count_parameters(base_model)
        report_lines.append(f"- **Model Name:** {BASE_MODEL}\n")
        report_lines.append(f"- **Total Parameters:** {base_total:,}\n")
        report_lines.append(f"- **Quantization:** 4-bit NormalFloat4\n")
        report_lines.append(f"- **Hidden Size:** {base_model.config.hidden_size}\n")
        report_lines.append(f"- **Num Layers:** {base_model.config.num_hidden_layers}\n")
        report_lines.append(f"- **Num Attention Heads:** {base_model.config.num_attention_heads}\n\n")
        
        del base_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠ Нүүр хэсгийн загвар ачаалахад алдаа: {e}")
        report_lines.append(f"⚠ Error loading base model: {e}\n\n")
    
    # ============================================================
    # 2. LoRA Adapter Тохиргоо
    # ============================================================
    print("[2/6] LoRA adapter тохиргоог шалгаж байна...")
    report_lines.append("## 2. LoRA Adapter Configuration\n")
    
    adapter_config = load_adapter_config()
    if adapter_config:
        report_lines.append(f"- **LoRA Rank (r):** {adapter_config.get('r', 'N/A')}\n")
        report_lines.append(f"- **LoRA Alpha:** {adapter_config.get('lora_alpha', 'N/A')}\n")
        report_lines.append(f"- **LoRA Dropout:** {adapter_config.get('lora_dropout', 'N/A')}\n")
        report_lines.append(f"- **Target Modules:** {', '.join(adapter_config.get('target_modules', []))}\n")
        report_lines.append(f"- **Task Type:** {adapter_config.get('task_type', 'N/A')}\n")
        report_lines.append(f"- **Inference Mode:** {adapter_config.get('inference_mode', 'N/A')}\n\n")
    
    # ============================================================
    # 3. Сургасан моделийг ачаалах
    # ============================================================
    print("[3/6] Сургасан stage1 загварыг ачаалж байна...")
    report_lines.append("## 3. Stage 1 Fine-tuned Model\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # LoRA adapter ачаалах
        if os.path.exists(STAGE1_MODEL_PATH):
            model = PeftModel.from_pretrained(model, STAGE1_MODEL_PATH)
            model.eval()
            report_lines.append(f"- **LoRA Adapter Path:** {STAGE1_MODEL_PATH}\n")
            report_lines.append(f"- **Status:** ✓ Successfully loaded\n\n")
        else:
            report_lines.append(f"⚠ Model path not found: {STAGE1_MODEL_PATH}\n\n")
            
    except Exception as e:
        print(f"⚠ Моделийг ачаалахад алдаа: {e}")
        report_lines.append(f"⚠ Error loading model: {e}\n\n")
        return
    
    # ============================================================
    # 4. Сургалтын түүхүүд
    # ============================================================
    print("[4/6] Сургалтын логуудыг уншиж авиж байна...")
    report_lines.append("## 4. Training History\n")
    
    training_logs = load_training_logs()
    if training_logs:
        # Үзүүлэлтүүдийг авах
        losses = [log.get('loss') for log in training_logs if 'loss' in log]
        learning_rates = [log.get('learning_rate') for log in training_logs if 'learning_rate' in log]
        
        report_lines.append(f"- **Total Steps:** {len(training_logs)}\n")
        if losses:
            report_lines.append(f"- **Final Loss:** {losses[-1]:.4f}\n")
            report_lines.append(f"- **Initial Loss:** {losses[0]:.4f}\n")
            report_lines.append(f"- **Min Loss:** {min(losses):.4f}\n")
        
        if learning_rates:
            report_lines.append(f"- **Learning Rate:** {learning_rates[0]}\n")
        
        # Цаг хугацаа
        if training_logs:
            first_log = training_logs[0]
            last_log = training_logs[-1]
            if 'epoch' in first_log and 'epoch' in last_log:
                report_lines.append(f"- **Epochs:** {last_log.get('epoch', 'N/A')}\n")
        
        report_lines.append("\n")
    
    training_summary = load_training_summary()
    if training_summary:
        report_lines.append("### Training Summary\n")
        report_lines.append(f"- **Training Loss:** {training_summary.get('train_loss', 'N/A')}\n")
        report_lines.append(f"- **Epochs Trained:** {training_summary.get('epoch', 'N/A')}\n")
        report_lines.append(f"- **Steps:** {training_summary.get('global_step', 'N/A')}\n\n")
    
    # ============================================================
    # 5. Моделийн үзүүлэлтүүдийг туршиж авах
    # ============================================================
    print("[5/6] Моделийг монгол хэлний текстүүдээр туршиж байна...")
    report_lines.append("## 5. Model Test Results (Mongolian Prompts)\n")
    
    report_lines.append("### Sample Generations\n\n")
    
    with torch.no_grad():
        for i, prompt in enumerate(MONGOLIAN_PROMPTS[:3], 1):
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                report_lines.append(f"**Prompt {i}:** {prompt}\n\n")
                report_lines.append(f"**Generated:** {generated_text}\n\n")
                
            except Exception as e:
                print(f"⚠ Үүсгэхэд алдаа (prompt {i}): {e}")
                report_lines.append(f"⚠ Error: {e}\n\n")
    
    # ============================================================
    # 6. Файлын асуулт
    # ============================================================
    print("[6/6] Моделийн файлуудыг шалгаж байна...")
    report_lines.append("## 6. Model Files\n")
    
    if os.path.exists(STAGE1_MODEL_PATH):
        for file in os.listdir(STAGE1_MODEL_PATH):
            file_path = os.path.join(STAGE1_MODEL_PATH, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                report_lines.append(f"- {file}: {format_size(size)}\n")
        report_lines.append("\n")
    
    # ============================================================
    # Тайланг хадгалах
    # ============================================================
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print("\n" + "="*70)
    print(f"✓ Тайлан амжилттай үүсгэлээ: {OUTPUT_REPORT}")
    print("="*70)
    
    return report_lines

if __name__ == "__main__":
    main()
