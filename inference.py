from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-2-13b-hf"
local_model = "./outputs/lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(
#     local_model, 
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map='auto'
# )
model = AutoModelForCausalLM.from_pretrained(model_id)

text = "What is the capital of France?"
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))