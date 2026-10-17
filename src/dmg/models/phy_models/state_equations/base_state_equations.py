from abc import ABC, abstractmethod
from collections.abc import Callable

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
    def build_rhs(
        self,
        theta: dict[str, torch.Tensor],
        forcing: torch.Tensor,
        *,
        t_start: torch.Tensor,
        dt: float,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the function defining the  r.h.s. of the differential equation.

        The NN predicts a new set of parameters for every batch, which changes
        the differential equation. This builder allows us to account for per-batch
        parameters, as well as the meteorological forcings for the batch.

        For torchdiffeq compatibility, the callable built by this method should only contain
        time and state as input parameters in the following order:

        def rhs(t, state):
            ...

        Arguments:
        ---
        theta: the set of parameters for the current batch
        forcing: the meteorological forcing data input(s) that needs to be used in the model.
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
