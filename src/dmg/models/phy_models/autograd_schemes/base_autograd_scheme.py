import torch

from dmg.models.phy_models.state_equations.base_state_equations import (
    BaseStateEquations,
)


class BaseAutogradScheme(torch.autograd.Function):
    """Base class for autograd strategies over a `BaseStateEquations`."""
    # E.g. discrete adjoint, continuous adjoint state, forward-sensitivity
    @staticmethod
    def forward(
        ctx,
        state_equations: BaseStateEquations,
        solver: dict,
        initial_state: torch.Tensor,
        forcing: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        theta_keys: tuple[str, ...],
        *theta_values: torch.Tensor,
    ) -> torch.Tensor:
        """Custom forward pass, with optional torchdiffeq solver keyword arguments.

        Implementations may call ``integrate_states`` for the state solve.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Custom backward pass."""
        raise NotImplementedError
