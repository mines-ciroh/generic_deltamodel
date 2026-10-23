"""Train LSTM + HBV process model with plain PyTorch autograd by default."""
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
PERIOD = 'TEST'
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
    gage_ids = np.load(config['observations']['gage_info'])

    with open(config['observations']['subset_path']) as subset_file:
        selected_basins = json.load(subset_file)
    basin_indices = np.intersect1d(
        selected_basins,
        gage_ids,
        return_indices=True,
    )[2]

    hydrograph(
        obs_path=os.path.join(config['sim_dir'], 'streamflow_obs.npy'),
        sim_path=os.path.join(config['sim_dir'], 'streamflow.npy'),
        output_dir=config['plot_dir'],
        start_time=config['test']['start_time'],
        basin_ids=gage_ids[basin_indices],
        warmup=config['model'].get('warmup', 0),
        output_unit=config['model'].get('output_unit', 'mm/day'),
    )
    
    print(f"Testing complete. Outputs saved to \n{out_dir}")


def main() -> None:
    """Dispatch to training or evaluation using the script settings."""
    if PERIOD == "TRAIN":
        train(CONFIG_PATH)
    elif PERIOD == "TEST":
        test(CONFIG_PATH, TEST_EPOCH)
    else:
        raise ValueError(f"Invalid PERIOD: {PERIOD}")


def hydrograph(
    obs_path: str,
    sim_path: str,
    output_dir: str,
    start_time: str,
    basin_ids: np.ndarray,
    warmup: int = 0,
    output_unit: str = 'mm/day',
) -> None:
    """Save one observed-versus-simulated hydrograph for each basin."""
    observations = np.load(obs_path)
    simulations = np.load(sim_path)

    if simulations.ndim == 3 and simulations.shape[-1] == 1:
        simulations = simulations[..., 0]
    if observations.ndim != 2 or simulations.ndim != 2:
        raise ValueError(
            'Expected streamflow observations and simulations with shape '
            '(time, basins).'
        )
    if observations.shape[1] != simulations.shape[1]:
        raise ValueError(
            f'Observation and simulation basin counts differ: '
            f'{observations.shape[1]} != {simulations.shape[1]}.'
        )
    if len(basin_ids) != simulations.shape[1]:
        raise ValueError(
            f'Basin ID count ({len(basin_ids)}) does not match the number of '
            f'simulated basins ({simulations.shape[1]}).'
        )

    time_difference = observations.shape[0] - simulations.shape[0]
    if time_difference == warmup:
        observations = observations[warmup:]
        first_timestep = pd.Timestamp(start_time) + pd.Timedelta(days=warmup)
    elif time_difference == 0:
        first_timestep = pd.Timestamp(start_time)
    else:
        raise ValueError(
            f'Cannot align {observations.shape[0]} observations with '
            f'{simulations.shape[0]} simulations using warmup={warmup}.'
        )

    timesteps = pd.date_range(first_timestep, periods=simulations.shape[0], freq='D')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for basin_index in range(simulations.shape[1]):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            timesteps,
            observations[:, basin_index],
            label='Observation',
            color='black',
        )
        ax.plot(
            timesteps,
            simulations[:, basin_index],
            label='Simulation',
            color='tab:red',
        )
        ax.set(
            title=f'Streamflow hydrograph - basin {basin_ids[basin_index]}',
            xlabel='Time (daily)',
            ylabel=f'Streamflow ({output_unit})',
        )
        ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_path / f'streamflow_basin_{basin_index + 1}.png', dpi=150)
        plt.close(fig)

if __name__ == '__main__':
    main()
