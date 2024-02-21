import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import transformers

def process_model(base_model, peft_model_id, save_path):
    print(f"base model: {base_model}")
    print(f"Processing model from: {peft_model_id}")
    print(f"Saving merged model to: {save_path}")
    print(transformers.__version__)

    pi_model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_model)

    model = PeftModel.from_pretrained(pi_model, peft_model_id)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(save_path)
    tok.save_pretrained(save_path)

if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    raise ValueError("This script expects two arguments: peft_model_id and save_path")

    base_model = sys.argv[1]
    peft_model_id = sys.argv[2]
    save_path = sys.argv[3]
    process_model(base_model, peft_model_id, save_path)