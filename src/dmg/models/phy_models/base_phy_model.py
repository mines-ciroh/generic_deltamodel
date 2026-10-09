"""Contract for composable physical models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from dmg.models.phy_models.autograd_schemes.base_autograd_scheme import (
    BaseAutogradScheme,
)
from dmg.models.phy_models.integration import (
    integrate_states,
    validate_integration_inputs,
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

    def _simulate(
        self,
        forcing: torch.Tensor,
        t: torch.Tensor,
        theta: dict[str, torch.Tensor],
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return states, streamflow."""
        if self.autograd_scheme is None:
            states = integrate_states(
                self.state_equations,
                forcing,
                t,
                theta,
                self.solver,
                self.dt,
                initial_state,
            )
        else:
            validate_integration_inputs(forcing, t, theta, self.solver, self.dt)
            states = self.autograd_scheme.apply(
                self.state_equations,
                self.solver,
                initial_state,
                forcing,
                t,
                self.dt,
                tuple(theta),
                *theta.values(),
            )  # [time, n_states, basins, nmul]

        previous_states = torch.cat((initial_state.unsqueeze(0), states[:-1]))

        streamflow = torch.stack(
            [
                self.state_equations.compute_streamflow(
                    t=t[i] + self.dt,
                    state=states[i],
                    previous_state=previous_states[i],
                    forcing=forcing[i],
                    theta={name: value[i] for name, value in theta.items()},
                )
                for i in range(t.numel())
            ]
        ).mean(dim=-1)

        return states, streamflow
