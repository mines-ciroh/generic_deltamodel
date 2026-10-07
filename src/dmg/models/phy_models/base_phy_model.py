"""Contract for composable physical models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from dmg.models.phy_models.autograd_schemes.base_autograd_scheme import (
    BaseAutogradScheme,
)
from dmg.models.phy_models.state_equations.base_state_equations import (
    BaseStateEquations,
)


class BasePhyModel(torch.nn.Module, ABC):
    """Base physical model."""

    state_equations: BaseStateEquations  # The physics (defines a vector field)
    solver: dict  # The torchdiffeq solver to use
    autograd_scheme: type[BaseAutogradScheme] | None  # None is default autograd

    @property
    @abstractmethod
    def learnable_param_count(self) -> int:
        """Number of NN parameters per timestep, including any ``nmul`` sets."""

    @abstractmethod
    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute predictions from model inputs and NN parameters.

        Parameters
        ----------
        data_dict
            Batch inputs and optional metadata. Required keys depend on the model.
        parameters
            Raw NN outputs shaped ``[input_time, batch_size, learnable_param_count]``.
            Parameter ordering and scaling depend on the model.

        Returns
        -------
        dict[str, torch.Tensor]
            Named predictions, typically shaped ``[prediction_time, batch_size, channels]``
            after warmup removal. Names, units, and time alignment depend on the model.
        """
