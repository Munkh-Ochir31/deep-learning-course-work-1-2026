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