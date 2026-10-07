from functools import partial

import torch
from torchdiffeq import odeint

from dmg.models.phy_models.state_equations.base_state_equations import (
    BaseStateEquations,
)


def integrate_states(
    state_equations: BaseStateEquations,
    forcing: torch.Tensor,
    t: torch.Tensor,
    theta: dict[str, torch.Tensor],
    solver: dict,
    dt: float = 1.0,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Wrapper for torchdiffeq.odeint() for f in StateEquations."""
    if initial_state is None:
        initial_state = state_equations.initialize_state(forcing)
    state = initial_state
    t = t.to(device=state.device, dtype=state.dtype)
    validate_integration_inputs(forcing, t, theta, solver, dt)

    states = []
    for i, t_i in enumerate(t):
        # Set the ODE for torchdiffeq ingestion
        f = partial(
            state_equations.f,
            forcing=forcing[i],
            theta={name: value[i] for name, value in theta.items()},
        )
        # step through f with the solver of choice and return the states.
        state = odeint(f, state, torch.stack((t_i, t_i + dt)), **solver)[-1]
        states.append(state)

    return torch.stack(states, dim=0)


def validate_integration_inputs(
    forcing: torch.Tensor,
    t: torch.Tensor,
    theta: dict[str, torch.Tensor],
    solver: dict,
    dt: float,
) -> None:
    """Validate interval inputs and require a gradient-preserving solver."""
    if dt <= 0:
        raise ValueError('dt must be positive.')
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError('t must be a nonempty one-dimensional tensor.')
    if forcing.shape[0] != t.numel() or any(
        value.shape[0] != t.numel() for value in theta.values()
    ):
        raise ValueError('Forcing and parameter time axes must match t.')
    if not torch.allclose(t[1:] - t[:-1], torch.full_like(t[1:], dt)):
        raise ValueError('Interval starts in t must be spaced by dt.')
    if solver.get('method') == 'scipy_solver':
        raise ValueError('scipy_solver does not preserve PyTorch gradients.')
