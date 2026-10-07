from abc import ABC, abstractmethod

import torch


class BaseStateEquations(ABC):
    """Base class for a process-based rainfall-runoff model's state equations.

    The state equations will follow the form (dS/dt = f(S, theta, x, t)).

    Where S are storage states, theta are model parameters,
    x are meteorological forcings, and t is time.

    Note: Static catchment attributes A are typically not applied in PBMs so we will not
    use them as a part of BaseStateEquations.

    BaseStateEquations implementations should contain the physics of the hydrology model as a
    continuous-time vector field, plus the streamflow readout and routing applied to its states.

    This decoupling allows us to separate the autograd scheme and the ODE solver from the
    physical equations.
    """

    @abstractmethod
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physically valid (min, max) ranges for each parameter in the model."""

    @abstractmethod
    def initialize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Build the initial state tensor, shaped ``[n_states, *batch_dims]``, from forcings x."""

    @abstractmethod
    def f(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        *,
        forcing: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return f=dS/dt (potentially a vector of state derivatives).

        Note: The position of `t` and `state` in the function arguments
        match torchdiffeq's API for odeint().
        """

    @abstractmethod
    def compute_streamflow(
        self,
        *,
        t: torch.Tensor,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        forcing: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the simulated streamflow over [t - dt, t] from the states at both ends.

        These values will then be routed, and then used to compare to the streamflow observations
        in the loss function.
        """

    @abstractmethod
    def route_streamflow(
        self,
        *,
        streamflow: torch.Tensor,
        **routing_params: torch.Tensor,
    ) -> torch.Tensor:
        """Smear the streamflow sequences to a single value to be used in the loss function.

        An example of this is the unit hydrograph.
        """
