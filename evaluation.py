import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path='./outputs/test2/',
    local_files_only=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print(model)