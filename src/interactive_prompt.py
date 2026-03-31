"""
Интерактив Prompt Тестер - Stage 1 Загварыг сругуулах
Статик загваруудын замыг өгөөд prompt-уудыг гараас оруулан үр дүнг үзэх
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# Статик Тохиргоо
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B"
STAGE1_MODEL_PATH = "models/stage1_pretrained"

# Сургалтын параметрүүд
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
DO_SAMPLE = True

# ============================================================
# Загварыг ачаалах
# ============================================================

def load_model():
    """Загварыг ачаалах"""
    print("=" * 70)
    print("Загварыг ачаалж байна...")
    print("=" * 70)
    
    # 4-bit quantization тохиргоо
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Токенайзер ачаалах
    print(f"[1/3] Токенайзер ачаалж байна: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Нүүр загвар ачаалах
    print(f"[2/3] Нүүр загвар ачаалж байна: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # LoRA адаптер ачаалах
    print(f"[3/3] LoRA адаптер ачаалж байна: {STAGE1_MODEL_PATH}")
    if os.path.exists(STAGE1_MODEL_PATH):
        model = PeftModel.from_pretrained(model, STAGE1_MODEL_PATH)
        print("\n✓ Загвар амжилттай ачаалагдлаа!")
    else:
        print(f"\n⚠ Анхаар: {STAGE1_MODEL_PATH} байршил олдоогүй байна!")
    
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print()
    
    return model, tokenizer

# ============================================================
# Текст үүсгэх функц
# ============================================================

def generate_text(model, tokenizer, prompt, max_tokens=MAX_NEW_TOKENS, 
                  temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    """Prompt-ийн суурьт текст үүсгэх"""
    
    try:
        # Prompt-ийг токенизировать хийх
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Текст үүсгэх
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Үр дүнг decoding хийх
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    except Exception as e:
        print(f"❌ Алдаа: {e}")
        return None

# ============================================================
# Интерактив Loop
# ============================================================

def main():
    # Загварыг ачаалах
    model, tokenizer = load_model()
    
    print("=" * 70)
    print("ИНТЕРАКТИВ PROMPT ТЕСТЕР")
    print("=" * 70)
    print("Prompt бичээд [Enter] дар. Гаралахын тулд 'exit' гэж бич.\n")
    
    iteration = 1
    
    while True:
        try:
            # Prompt авах
            prompt = input(f"[{iteration}] Prompt: ").strip()
            
            # Гаралтын шалгалт
            if prompt.lower() == "exit":
                print("\nПрограм дуусч байна...")
                break
            
            if not prompt:
                print("⚠ Хоосон prompt байна. Дахин оруулна уу.\n")
                continue
            
            # Текст үүсгэх
            print("\n[Уусгаж байна...]")
            generated = generate_text(model, tokenizer, prompt)
            
            if generated:
                print(f"\n{'='*70}")
                print(f"ҮҮСГЭСЭН ТЕКСТ:\n")
                print(generated)
                print(f"\n{'='*70}\n")
            
            iteration += 1
        
        except KeyboardInterrupt:
            print("\n\nПрограм дуусч байна...")
            break
        except Exception as e:
            print(f"❌ Алдаа: {e}\n")

# ============================================================
# Ашигласан параметрүүдийг үзүүлэх
# ============================================================

def print_config():
    """Тохиргоог үзүүлэх"""
    print("\n📋 Загварын тохиргоо:")
    print(f"  - Base Model: {BASE_MODEL}")
    print(f"  - Stage 1 Model: {STAGE1_MODEL_PATH}")
    print(f"  - Max Tokens: {MAX_NEW_TOKENS}")
    print(f"  - Temperature: {TEMPERATURE}")
    print(f"  - Top P: {TOP_P}")
    print(f"  - Top K: {TOP_K}")
    print()

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    print_config()
    main()
