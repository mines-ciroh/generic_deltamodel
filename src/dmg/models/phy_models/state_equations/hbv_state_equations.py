import torch
from hydrodl2.core.calc import uh_conv, uh_gamma

from .base_state_equations import BaseStateEquations


class HbvStateEquations(BaseStateEquations):
    """State equations for the HBV hydrological model.

    Adapted from `HBV.forward` in hydrodl2's `hbv_adj.py` (Song et al., 2024).
    """

    n_stores = 5

    # Static routing parameters, kept separate from the per-timestep physical parameters.
    routing_parameter_bounds = {
        'rout_a': (0.0, 2.9),
        'rout_b': (0.0, 6.5),
    }

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """The bounds for each of HBV's parameters, in hbv_adj's order."""
        return {
            'beta': (1.0, 6.0),
            'fc': (50, 1000),
            'k0': (0.05, 0.9),
            'k1': (0.01, 0.5),
            'k2': (0.001, 0.2),
            'lp': (0.2, 1),
            'perc': (0, 10),
            'uzl': (0, 100),
            'tt': (-2.5, 2.5),
            'dd': (0.5, 10),
            'rfz': (0, 0.1),
            'cwh': (0, 0.2),
            'gamma': (0.3, 5),
        }

    def initialize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize five stores and a cumulative streamflow state."""
        n_states = self.n_stores + 1
        return torch.zeros(n_states, *x.shape[1:-1], dtype=x.dtype, device=x.device)

    def f(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        *,
        forcing: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Stack the ODEs into dS/dt for [snow, meltwater, soil, upper_zone, lower_zone, Q]."""
        fluxes = self._fluxes(state=state, forcing=forcing, theta=theta)

        d_snow_dt = fluxes['snowfall'] + fluxes['refreeze'] - fluxes['melt']
        d_meltwater_dt = fluxes['melt'] - fluxes['refreeze'] - fluxes['to_soil']
        d_soil_dt = (
            fluxes['to_soil'] + fluxes['rainfall'] - fluxes['recharge'] - fluxes['excess'] - fluxes['et']
        )
        d_upper_zone_dt = (
            fluxes['recharge'] + fluxes['excess'] - fluxes['perc'] - fluxes['q0'] - fluxes['q1']
        )
        d_lower_zone_dt = fluxes['perc'] - fluxes['q2']
        d_streamflow_dt = fluxes['q0'] + fluxes['q1'] + fluxes['q2']

        return torch.stack((
            d_snow_dt,
            d_meltwater_dt,
            d_soil_dt,
            d_upper_zone_dt,
            d_lower_zone_dt,
            d_streamflow_dt,
        ))

    def compute_streamflow(
        self,
        *,
        t: torch.Tensor,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        forcing: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Unrouted streamflow Q = Q0 + Q1 + Q2 integrated over the interval [t - dt, t]."""
        return state[self.n_stores] - previous_state[self.n_stores]

    def route_streamflow(
        self,
        *,
        streamflow: torch.Tensor,
        rout_a: torch.Tensor,
        rout_b: torch.Tensor,
        len_uh: int = 15,
    ) -> torch.Tensor:
        """Route streamflow [time, basins] with a gamma unit hydrograph.

        `rout_a` and `rout_b` are dynamic descaled routing parameters of shape [basins].
        """
        n_steps = streamflow.shape[0]
        a = rout_a.unsqueeze(0).repeat(n_steps, 1).unsqueeze(-1)
        b = rout_b.unsqueeze(0).repeat(n_steps, 1).unsqueeze(-1)
        uh = uh_gamma(a, b, lenF=len_uh).permute(1, 2, 0)  # [basins, 1, len_uh]
        rf = streamflow.unsqueeze(-1).permute(1, 2, 0)  # [basins, 1, time]
        return uh_conv(rf, uh).permute(2, 0, 1).squeeze(-1)

    def _fluxes(
        self,
        *,
        state: torch.Tensor,
        forcing: torch.Tensor,
        theta: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate every HBV flux at the given state."""
        snowpack, meltwater, soil, upper_zone, lower_zone = state[:5].unbind(dim=0)
        snowpack = snowpack.clamp(min=0.0)
        meltwater = meltwater.clamp(min=0.0)
        soil = soil.clamp(min=0.0)
        upper_zone = upper_zone.clamp(min=0.0)
        lower_zone = lower_zone.clamp(min=0.0)

        P, T, Ep = forcing.unbind(dim=-1)

        beta, fc, lp = theta['beta'], theta['fc'], theta['lp']
        tt, cfmax, cfr, cwh = theta['tt'], theta['dd'], theta['rfz'], theta['cwh']

        snowfall = P * (T < tt)
        rainfall = P * (T >= tt)
        refreeze = torch.minimum(
            (cfr * cfmax * (tt - T)).clamp(min=0.0), meltwater
        )
        melt = torch.minimum((cfmax * (T - tt)).clamp(min=0.0), snowpack)
        to_soil = (meltwater - cwh * snowpack).clamp(min=0.0)


        soil_wetness = ((soil / fc).clamp(min=1e-8) ** beta).clamp(max=1.0)
        recharge = (rainfall + to_soil) * soil_wetness
        excess = (soil - fc).clamp(min=0.0)
        evap_factor = ((soil / (lp * fc)).clamp(min=1e-8) ** theta['gamma']).clamp(max=1.0)
        et = torch.minimum(soil, Ep * evap_factor)

        perc = torch.minimum(upper_zone, theta['perc'])
        q0 = theta['k0'] * (upper_zone - theta['uzl']).clamp(min=0.0)
        q1 = theta['k1'] * upper_zone
        q2 = theta['k2'] * lower_zone

        return {
            'snowfall': snowfall,
            'rainfall': rainfall,
            'refreeze': refreeze,
            'melt': melt,
            'to_soil': to_soil,
            'recharge': recharge,
            'excess': excess,
            'et': et,
            'perc': perc,
            'q0': q0,
            'q1': q1,
            'q2': q2,
        }
