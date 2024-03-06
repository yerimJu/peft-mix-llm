python finetune.py \
    --base_model 'meta-llama/Llama-2-13b-hf' \
    --data_path './alpaca_data_gpt4_700.json' \
    --output_dir './outputs/lora' \
    --batch_size 16 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 200 \
    --peft_method lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --wandb_project 'id-data-descriptor-llm' \
    --wandb_run_name 'llama2-sft-lora'
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/qlora' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method qlora \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --lora_target_modules '[q_proj,v_proj]' \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-qlora'
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/adalora' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method adalora \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-adalora' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/ia3' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method ia3 \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-ia3' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/loha' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method loha \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-loha' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/lokr' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method lokr \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-lokr' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/prefix_tuning' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method prefix_tuning \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-prefix_tuning' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/prompt_tuning' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method prompt_tuning \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-prompt_tuning' && \
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/vera' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method vera \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-vera'
# python finetune.py \
#     --base_model 'meta-llama/Llama-2-13b-hf' \
#     --data_path './alpaca_data_gpt4_700.json' \
#     --output_dir './outputs/dora' \
#     --batch_size 16 \
#     --micro_batch_size 2 \
#     --num_epochs 3 \
#     --learning_rate 1e-4 \
#     --cutoff_len 512 \
#     --val_set_size 200 \
#     --peft_method dora \
#     --train_on_inputs \
#     --group_by_length \
#     --wandb_project 'id-data-descriptor-llm' \
#     --wandb_run_name 'llama2-sft-dora'