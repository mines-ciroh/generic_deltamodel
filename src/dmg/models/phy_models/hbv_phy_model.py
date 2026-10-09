import torch

from dmg.core.utils.factory import import_autograd_scheme
from dmg.models.phy_models.base_phy_model import BasePhyModel
from dmg.models.phy_models.state_equations.hbv_state_equations import HbvStateEquations


class HbvPhyModel(BasePhyModel):
    """The HBV physical rainfall-runoff model.

    The StateEquations for this model are adapted from hydrodl2's HbvAdj.
    """

    state_equations = HbvStateEquations()
    dt = 1.0  # Forcing interval length in days.

    def __init__(self, config: dict, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.warmup = config.get('warmup') or 0
        self.nmul = config.get('nmul') or 1  # The number of parameter sets to use.
        self.dy_drop = config.get('dy_drop') or 0.0
        self.dynamic_params = (config.get('dynamic_params') or {}).get(
            type(self).__name__, []
        )

        # Instantiate the autograd scheme from the config, otherwise use default pytorch autograd
        autograd_scheme = config.get('autograd_scheme')
        self.autograd_scheme = (
            import_autograd_scheme(autograd_scheme) if autograd_scheme else None
        )

        # Instantiate the torchdiffeq solver from the config, otherwise use the default (euler)
        solver = config.get('solver')
        self.solver = {'method': 'euler', 'rtol': 1e-4, 'atol': 1e-6}
        self.solver.update(solver or {})
        self.solver['options'] = dict(self.solver.get('options') or {})
        if self.solver['method'] == 'euler':
            # Combined upper-zone withdrawal can reach 2.4 * storage per day.
            self.solver['options'].setdefault('step_size', 0.25)

    @property
    def learnable_param_count(self) -> int:
        """# physical parameters * number of ensembles + # static routing parameters."""
        return len(self.state_equations.parameter_bounds()) * self.nmul + len(
            self.state_equations.routing_parameter_bounds
        )

    def forward(
        self,
        data_dict: dict,
        parameters: torch.Tensor,  # [time, basins, #params], raw NN output
    ) -> dict:
        """Prepare parameters, simulate with warmup, and route daily streamflow."""
        x = data_dict['x_phy']
        n_timesteps = x.shape[0]

        if not 0 <= self.warmup < n_timesteps:
            raise ValueError('warmup must satisfy 0 <= warmup < input length.')
        if parameters.shape != (
            n_timesteps, x.shape[1], self.learnable_param_count
        ):
            raise ValueError('Parameter shape must match input time, basins, and count.')

        theta, routing_theta = self._prepare_parameters(parameters)

        forcing = x.unsqueeze(2).expand(-1, -1, self.nmul, -1)

        # timesteps for the ODE solve
        t = torch.arange(n_timesteps, device=x.device, dtype=x.dtype) * self.dt

        w = self.warmup
        initial_state = self.state_equations.initialize_state(forcing)
        warmup_streamflow = forcing.new_zeros((0, x.shape[1]))
        if w > 0:
            with torch.no_grad():
                # compute warmup states and streamflow
                warmup_states, warmup_streamflow = self._simulate(
                    forcing[:w],
                    t[:w],
                    {name: value[:w] for name, value in theta.items()},
                    initial_state,
                )
            initial_state = warmup_states[-1]

        # compute prediction period states and streamflow from end of warmup
        _, prediction_streamflow = self._simulate(
            forcing[w:],
            t[w:],
            {name: value[w:] for name, value in theta.items()},
            initial_state,
        )

        streamflow = torch.cat((warmup_streamflow, prediction_streamflow))
        routed = self.state_equations.route_streamflow(
            streamflow=streamflow,
            **routing_theta,
        )

        return {'streamflow': routed[self.warmup:].unsqueeze(-1)}


    def _make_phy_parameters(
        self,
        phy_params: torch.Tensor,
        dynamic_params: list[str],
    ) -> torch.Tensor:
        """Hold parameters static at the last timestep, except those in `dynamic_params`.
        
        TODO: revisit this
        """
        static = phy_params[-1:].expand_as(phy_params)
        if not dynamic_params:
            return static

        names = list(self.state_equations.parameter_bounds())
        is_dynamic = torch.tensor(
            [name in dynamic_params for name in names], device=phy_params.device
        ).unsqueeze(-1)
        # dy_drop randomly reverts dynamic params to static, per basin, parameter, and component.
        drop = torch.zeros_like(phy_params[:1], dtype=torch.bool)
        if self.training and self.dy_drop > 0:
            drop = torch.bernoulli(
                torch.full(
                    (1, *phy_params.shape[1:]), self.dy_drop, device=phy_params.device
                )
            ).bool()
        return torch.where(is_dynamic & ~drop, phy_params, static)

    def _prepare_parameters(
        self, parameters: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Prepare parameters for simulation."""
        n_steps, n_basins, _ = parameters.shape
        bounds = self.state_equations.parameter_bounds()
        n_phy = len(bounds)

        # bound the raw NN output parameters to (0,1)
        phy_params = torch.sigmoid(parameters[..., : n_phy * self.nmul]).view(
            n_steps, n_basins, n_phy, self.nmul
        )
        rout_params = torch.sigmoid(parameters[-1, :, n_phy * self.nmul :])

        # Warmup and prediction periods each hold static params at their own last timestep.
        prediction_params = self._make_phy_parameters(
            phy_params[self.warmup :], self.dynamic_params
        )

        # Combine the static warmup parameters with the static prediction parameters.
        if self.warmup > 0:
            warmup_params = self._make_phy_parameters(phy_params[: self.warmup], [])
            phy_params = torch.cat((warmup_params, prediction_params), dim=0)
        else:
            phy_params = prediction_params

        # Descale the static & dynamic parameters
        theta = {
            name: lo + phy_params[:, :, i] * (hi - lo)
            for i, (name, (lo, hi)) in enumerate(bounds.items())
        }  # each [time, basins, nmul]

        routing_theta = {
            name: lo + rout_params[:, i] * (hi - lo)
            for i, (name, (lo, hi)) in enumerate(
                self.state_equations.routing_parameter_bounds.items()
            )
        }
        return theta, routing_theta
