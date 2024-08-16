from src import run

args = {
    "stage": "sft",
    "do_train": True,
    # "predict_with_generate": True,
    "template": "chatglm3",
    "model_name_or_path": "models/roborock_control_lora_sfsfsf_merged",
    # "cache_path": "/root/PycharmProjects/ChatGLM3/finetune_chatmodel_demo/formatted_data/hf_dataset/tool_alpaca.hf",
    "dataset": "roborock_control",
    "cutoff_len": 2048,
    "finetuning_type": "lora",
    "lora_target": "query_key_value",
    "output_dir": "models/sfsfsfsf",
    # "adapter_name_or_path": "adgen_lora/checkpoint-6000",
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 500,
    "learning_rate": 1e-3,
    "max_steps": 1000,
    "lora_rank": 32,
    "plot_loss": True,
    "fp16": True
}

run.main(args)