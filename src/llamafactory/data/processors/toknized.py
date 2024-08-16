def preprocess_tokenized_dataset(dataset, training_args, data_args):
    def preprocess_func(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        model_inputs = {'input_ids': [], 'labels': []}
        for item in examples['conversations']:
            data = json.loads(item)
            model_inputs['input_ids'].append(data['input_ids'])
            model_inputs['labels'].append(data['labels'])
            # model_inputs["attention_mask"].append([1] * len(data['input_ids']))
        return model_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache),
                desc="loading from tokenized dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,
            remove_columns=column_names,
            **kwargs
        )

        return dataset
