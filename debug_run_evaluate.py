from src import run_evaluate

args = {
    "stage": "sft",
    "do_predict": True,
    "predict_with_generate": True,
    "template": "chatglm3",
    "model_name_or_path": "/data/dataset/huggingface/hub/chatglm3-6b",
    "dataset": "roborock_control_testset",
    "finetuning_type": "lora",
    "lora_target": "query_key_value",
    "output_dir": "outputs/roborock_control_lora_output_finetune_dev",
    "adapter_name_or_path": "models/roborock_control_lora/checkpoint-1000",
    "per_device_eval_batch_size": 4,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "fp16": True,
    "cutoff_len": 2048,
    "max_samples": 100
    }

run_evaluate.main(args)