from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import transformers
import numpy as np
from utils.prompter import Prompter
import torch

import os

# to avoid out of memory
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# torch.cuda.empty_cache()

peft_method = "vera"
model_id = "meta-llama/Llama-2-13b-hf"
local_model = f"./outputs/{peft_method}"
os.environ["WANDB_PROJECT"] = "id-data-descriptor-llm"
wandb_run_name = f"gpt3.5-sft-{peft_method}"

data_path = './alpaca_data_gpt4_700.json'
output_dir = f'./outputs/{peft_method}_2'
val_set_size = 200
cutoff_len = 256
batch_size = 16
micro_batch_size = 1
prompt_template_name = 'alpaca'

prompter = Prompter(prompt_template_name)
bleu = evaluate.load("bleu")

def compute_metrics(eval_preds):
    print("compute_metrics()")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # skip label_id -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # preprocessing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    print(result)
    return result

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        local_model, 
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
        
    gradient_accumulation_steps = batch_size // micro_batch_size
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            report_to="wandb",
            run_name=wandb_run_name
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    print("trainer defined...")
    with torch.no_grad():
        print("start evaluation")
        results = trainer.evaluate()
        print(results)