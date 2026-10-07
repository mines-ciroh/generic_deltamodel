"""Train LSTM + HBV process model with plain PyTorch autograd by default."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example import load_config

from dmg import ModelHandler
from dmg.core.utils import (
    import_data_loader,
    import_trainer,
    print_config,
    set_randomseed,
)

# ------------------------------------------#
# Define model settings here.
CONFIG_PATH = '../example/conf/config_hbv_phy_model.yaml'
PERIOD = 'TRAIN'
TEST_EPOCH = 25
# ------------------------------------------#


def train(config_path: str) -> None:
    """Train the HBV process model using the given configuration."""
    config = load_config(config_path)
    config['mode'] = 'train'
    print_config(config)

    # Set random seed for reproducibility.
    set_randomseed(config['seed'])

    model = ModelHandler(config, verbose=True)

    data_loader_cls = import_data_loader(config['data_loader'])
    data_loader = data_loader_cls(config, test_split=True, overwrite=False)

    trainer_cls = import_trainer(config['trainer'])
    trainer = trainer_cls(
        config,
        model,
        train_dataset=data_loader.train_dataset,
    )

    trainer.train()
    print(f"Training complete. Model saved to \n{config['model_dir']}")


def test(config_path: str, test_epoch: int) -> None:
    """Evaluate the HBV process model at a specified epoch."""
    config = load_config(config_path)
    config['mode'] = 'test'
    config['test']['test_epoch'] = test_epoch

    # Keep test outputs next to the epochs instead of overwriting ./output/.
    out_dir = os.path.join(config['model_dir'], f'test_ep{test_epoch}/')
    config['output_dir'] = out_dir
    config['sim_dir'] = os.path.join(out_dir, 'sim/')
    config['plot_dir'] = os.path.join(out_dir, 'plot/')
    os.makedirs(config['sim_dir'], exist_ok=True)
    print_config(config)

    set_randomseed(config['seed'])

    model = ModelHandler(config, verbose=True)

    data_loader_cls = import_data_loader(config['data_loader'])
    data_loader = data_loader_cls(config, test_split=True, overwrite=False)

    trainer_cls = import_trainer(config['trainer'])
    trainer = trainer_cls(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
    )

    trainer.evaluate()
    print(f"Testing complete. Outputs saved to \n{out_dir}")


def main() -> None:
    """Dispatch to training or evaluation using the script settings."""
    if PERIOD == "TRAIN":
        train(CONFIG_PATH)
    elif PERIOD == "TEST":
        test(CONFIG_PATH, TEST_EPOCH)
    else:
        raise ValueError(f"Invalid PERIOD: {PERIOD}")


if __name__ == '__main__':
    main()
