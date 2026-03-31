# Stage 1 Model Inspection Report
**Generated on:** 2026-03-31 23:27:02

## 1. Base Model Information
- **Model Name:** Qwen/Qwen2.5-3B
- **Total Parameters:** 1,698,672,640
- **Quantization:** 4-bit NormalFloat4
- **Hidden Size:** 2048
- **Num Layers:** 36
- **Num Attention Heads:** 16

## 2. LoRA Adapter Configuration
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.1
- **Target Modules:** down_proj, o_proj, k_proj, up_proj, q_proj, v_proj, gate_proj
- **Task Type:** CAUSAL_LM
- **Inference Mode:** True

## 3. Stage 1 Fine-tuned Model
- **LoRA Adapter Path:** models/stage1_pretrained
- **Status:** ✓ Successfully loaded

## 4. Training History
- **Total Steps:** 612

### Training Summary
- **Training Loss:** N/A
- **Epochs Trained:** N/A
- **Steps:** N/A

## 5. Model Test Results (Mongolian Prompts)
### Sample Generations

**Prompt 1:** Монгол улсын нийслэл нь

**Generated:** Монгол улсын нийслэл нь Оросын нийслэл Хөвсгөл хотод орших Хөвсгөл аймгийн ИхУул сумын нутаг дэвсгэр

**Prompt 2:** Хиймэл оюун ухаан гэж

**Generated:** Хиймэл оюун ухаан гэж юу вэ Үндэсний мэдээллийн портал сайт . Хиймэл оюун ухаан гэж юу вэ оны р сар нд ц

**Prompt 3:** Монгол хэл дээр сайн байна гэж үнэ

**Generated:** Монгол хэл дээр сайн байна гэж үнэхээр хэлж болно. Тэр байтугай ард түмний багш нарын ажил хийж байгаад бид танай бичгийг

## 6. Model Files
- chat_template.jinja: 2.37 KB
- tokenizer_config.json: 668.00 B
- README.md: 5.06 KB
- tokenizer.json: 10.89 MB
- adapter_config.json: 1.02 KB
- training_args.bin: 5.08 KB
- adapter_model.safetensors: 114.25 MB

