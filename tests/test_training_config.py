from train.trainer import TrainingConfig


def test_training_config_defaults():
    cfg = TrainingConfig(model_name="gpt2", dataset_name="ds", train_split="train")
    assert cfg.grad_accum == 2
    assert cfg.lr == 5e-5
