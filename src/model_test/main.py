# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_path = r"C:\Users\my tech\Documents\3. 3B\deep-learning-course-work-1-2026\models\Qwen3.5-0.8B-Base"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# prompt = "Монгол хэл дээр AI гэж юу вэ?"

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=100
# )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig                                                   
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
                                                                    
# Текст үүсгэх                                                       
prompt = "Монгол хэл дээр AI гэж юу вэ?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)     
outputs = model.generate(**inputs, max_new_tokens=100)
print("Хариулт:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))