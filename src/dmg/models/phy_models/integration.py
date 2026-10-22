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

    # Check for errors before passing inputs to the solver
    validate_integration_inputs(forcing, t, theta, solver, dt)
    
    rhs = state_equations.build_rhs(theta, forcing, t_start=t[0], dt=dt)

    # Compiles the RHS for the repeated computations in odeint().
    rhs = torch.compile(
        rhs,
        fullgraph=True,
    )

    evaluation_times = torch.cat((t[:1], t+dt))

    return odeint(rhs, state, evaluation_times, **solver)[1:]


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
