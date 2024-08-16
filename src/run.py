from llmtuner import run_exp


def main(args):
    run_exp(args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(args)


if __name__ == "__main__":
    args = {
        "stage": "sft",
        "do_train": True,
        # "predict_with_generate": True,
        "template": "default",
        "model_name_or_path": "/data/dataset/modelscope/Yi-6B",
        # "model_name_or_path": "/data/dataset/huggingface/hub/chatglm3-6b",
        # "cache_path": "/root/PycharmProjects/ChatGLM3/finetune_chatmodel_demo/formatted_data/hf_dataset/tool_alpaca.hf",
        "dataset": "roborock_control_high_ppl",
        "flash_attn": True,
        "cutoff_len": 2048,
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": "models/roborock_control_lora_Yi_high_ppl",
        "adapter_name_or_path": "models/roborock_control_lora_Yi/checkpoint-125",
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 50,
        "learning_rate": 1e-3,
        "max_steps": 100,
        "lora_rank": 32,
        "plot_loss": True,
        "fp16": True
    }
    main(args)