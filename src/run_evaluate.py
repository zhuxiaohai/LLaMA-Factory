from llmtuner import run_exp


def main(args):
    run_exp(args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(args)


if __name__ == "__main__":
    args = {
        "stage": "sft",
        "do_predict": True,
        "predict_with_generate": True,
        "template": "default",
        "model_name_or_path": "/data/dataset/modelscope/Yi-6B",
        "dataset": "roborock_control_testset",
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": "outputs/roborock_control_lora_Yi_high_ppl_test_100",
        "adapter_name_or_path": "models/roborock_control_lora_Yi/checkpoint-125,models/roborock_control_lora_Yi_high_ppl/checkpoint-100",
        "per_device_eval_batch_size": 4,
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "fp16": True,
        "cutoff_len": 2048,
        "max_samples": 100
    }
    main(args)