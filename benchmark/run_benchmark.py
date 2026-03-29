import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch                                                                    
                  
bnb_config = BitsAndBytesConfig(                                                
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                                                  
    bnb_4bit_compute_dtype=torch.float16,
)                                                                               
                
model_name = "Qwen/Qwen2.5-3B"                                                  

tokenizer = AutoTokenizer.from_pretrained(model_name)                           
model = AutoModelForCausalLM.from_pretrained(
    model_name,                                                                 
    quantization_config=bnb_config,
    device_map="auto",                                                          
)               

print("Загвар амжилттай ачааллаа!")                                             
print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

data = []
path_jsonl = "/home/tr1bo/Documents/1. School/1. 3B/deep-learning-course-work-1-2026/benchmark/mongolian_llm_benchmark.jsonl"

with open(path_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))
print(f"Нийт асуулт: {len(data)}")


def make_prompt(item):
    task = item["task"]
    if task == "mcq":
        choices = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])])
        return f"Асуулт: {item['question']}\n{choices}\nЗөвхөн хариултыг бич.\nХариулт:"
    elif task == "reading":
        return f"Контекст: {item['context']}\nАсуулт: {item['question']}\nЗөвхөн хариултыг бич.\nХариулт:"
    elif task == "tool_calling":
        return f"Хэрэглэгчийн хүсэлт: {item['question']}\nJSON функц дуудалт:"
    else:
        return f"Асуулт: {item['question']}\nЗөвхөн хариултыг бич.\nХариулт:"


results = []
for i, item in enumerate(data):
    prompt = make_prompt(item)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    correct = item["answer"].lower().strip()
    is_correct = correct in generated.lower()

    results.append({
        "id": item["id"],
        "task": item["task"],
        "expected": item["answer"],
        "generated": generated,
        "correct": is_correct,
    })
    print(f"[{i+1}/{len(data)}] {item['task']} - {'O' if is_correct else 'X'} | {generated[:60]}")


import os
os.makedirs("results", exist_ok=True)

task_scores = {}
for task in set(r["task"] for r in results):
    task_results = [r for r in results if r["task"] == task]
    correct_count = sum(1 for r in task_results if r["correct"])
    task_scores[task] = {
        "correct": correct_count,
        "total": len(task_results),
        "accuracy": round(correct_count / len(task_results) * 100, 2)
    }

total_correct = sum(s["correct"] for s in task_scores.values())
total_questions = sum(s["total"] for s in task_scores.values())

output_data = {
    "model": "Qwen2.5-3B-base",
    "total_accuracy": round(total_correct / total_questions * 100, 2),
    "scores": task_scores,
    "details": results
}

with open("results/base_model_results.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("\n===== Үр дүн =====")
for task, score in sorted(task_scores.items()):
    print(f"  {task:15s}: {score['correct']}/{score['total']} ({score['accuracy']}%)")
print(f"  {'НИЙТ':15s}: {total_correct}/{total_questions} ({output_data['total_accuracy']}%)")
print(f"\nХадгалсан: results/base_model_results.json")