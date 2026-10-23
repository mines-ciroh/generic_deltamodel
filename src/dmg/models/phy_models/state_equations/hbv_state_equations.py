from typing import Literal

import softtorch as st
import torch
from hydrodl2.core.calc import uh_conv, uh_gamma

from .base_state_equations import BaseStateEquations


class HbvStateEquations(BaseStateEquations):
    """State equations for the HBV hydrological model.

    Adapted from `HBV.forward` in hydrodl2's `hbv_adj.py` (Song et al., 2024).

    Hard thresholds are relaxed with softtorch so that gradients do not vanish:

    Parameters
    ----------
    temperature_softness
        Softness of temperature thresholds (deg C).
    storage_softness
        Softness of storage/flux thresholds (mm or mm/day).
    ratio_softness
        Softness of dimensionless saturation ratios.
    mode
        softtorch relaxation family.
    """

    n_stores = 5

    def __init__(
        self,
        *,
        temperature_softness: float = 0.25,
        storage_softness: float = 0.1,
        ratio_softness: float = 0.01,
        mode: Literal['smooth', 'c0', 'c1', 'c2'] = 'smooth',
    ) -> None:
        for name, value in (
            ('temperature_softness', temperature_softness),
            ('storage_softness', storage_softness),
            ('ratio_softness', ratio_softness),
        ):
            if value <= 0:
                raise ValueError(f'{name} must be positive.')
        self.temperature_softness = temperature_softness
        self.storage_softness = storage_softness
        self.ratio_softness = ratio_softness
        self.mode = mode

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

    def build_rhs(
        self,
        theta: dict[str, torch.Tensor],
        forcing: torch.Tensor,
        *,
        t_start: torch.Tensor,
        dt: float,
    ):
        """Build the rhs of the ode with the current forcings and parameters."""
        if dt <= 0:
            raise ValueError('dt must be positive.')
        if forcing.shape[0] == 0:
            raise ValueError('Forcing must contain at least one timestep.')

        n_times = forcing.shape[0]

        def rhs(
            t: torch.Tensor,  # the timesteps to evaluate the ODE on
            state: torch.Tensor,
        ) -> torch.Tensor:
            """Stack the ODEs into dS/dt for [snow, meltwater, soil, upper_zone, lower_zone, Q].

            The function signature needs to be compatible with torchdiffeq.odeint()
            """
            # Linearly interpolate parameters/forcings.
            u = ((t - t_start) / dt).clamp(0, n_times - 1)
            i = u.floor().long()
            j = (i + 1).clamp(max=n_times - 1)
            weight = u - i

            i_index = i.reshape(1)
            j_index = j.reshape(1)

            # linear interpolate the forcings for the current times
            forcing_t = torch.lerp(
                forcing.index_select(0, i_index).squeeze(0),
                forcing.index_select(0, j_index).squeeze(0),
                weight,
            )

            # linear interpolate the parameters for the current times
            theta_t = {
                name: torch.lerp(
                    values.index_select(0, i_index).squeeze(0),
                    values.index_select(0, j_index).squeeze(0),
                    weight,
                )
                for name, values in theta.items()
            }

            fluxes = self._fluxes(state=state, forcing=forcing_t, theta=theta_t)

            d_snow_dt = fluxes['snowfall'] + fluxes['refreeze'] - fluxes['melt']
            d_meltwater_dt = fluxes['melt'] - fluxes['refreeze'] - fluxes['to_soil']
            d_soil_dt = (
                fluxes['to_soil']
                + fluxes['rainfall']
                - fluxes['recharge']
                - fluxes['excess']
                - fluxes['et']
            )
            d_upper_zone_dt = (
                fluxes['recharge']
                + fluxes['excess']
                - fluxes['perc']
                - fluxes['q0']
                - fluxes['q1']
            )
            d_lower_zone_dt = fluxes['perc'] - fluxes['q2']
            d_streamflow_dt = fluxes['q0'] + fluxes['q1'] + fluxes['q2']

            return torch.stack(
                (
                    d_snow_dt,
                    d_meltwater_dt,
                    d_soil_dt,
                    d_upper_zone_dt,
                    d_lower_zone_dt,
                    d_streamflow_dt,
                )
            )

        return rhs

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

        `rout_a` and `rout_b` are routing parameters of shape [basins].
        """
        # TODO: revisit this.
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
        P, T, Ep = forcing.unbind(dim=-1)  # Unpack forcings

        # Unpack parameters
        beta, fc, lp = theta['beta'], theta['fc'], theta['lp']
        tt, cfmax, cfr, cwh = theta['tt'], theta['dd'], theta['rfz'], theta['cwh']

        # Unpack storages and apply st.relu() to ensure no negative storages.
        # note: st = straight through, not softtorch.
        snowpack = self._soft_relu_st(state[0], softness=self.storage_softness)
        meltwater = self._soft_relu_st(state[1], softness=self.storage_softness)
        soil = self._soft_relu_st(state[2], softness=self.storage_softness)
        upper_zone = self._soft_relu_st(state[3], softness=self.storage_softness)
        lower_zone = self._soft_relu_st(state[4], softness=self.storage_softness)

        # Snow module
        # Grab the fraction of precipitation as snow vs. rain with st.less()
        snow_fraction = st.less(
            T, tt, softness=self.temperature_softness, mode=self.mode
        )
        rain_fraction = 1 - snow_fraction
        snowfall = P * snow_fraction
        rainfall = P * rain_fraction

        refreeze = self._soft_min_st(
            cfr * cfmax * self._soft_relu_st(tt - T, softness=self.temperature_softness),
            meltwater,
            softness=self.storage_softness,
        )

        melt = self._soft_min_st(
            cfmax * self._soft_relu_st(T - tt, softness=self.temperature_softness),
            snowpack,
            softness=self.storage_softness,
        )

        to_soil = self._soft_relu_st(
            meltwater - cwh * snowpack,
            softness=self.storage_softness,
        )

        # Soil moisture and evapotranspiration (ET) module
        soil_ratio = st.clamp(
            soil / fc, 0.0, 1.0, softness=self.ratio_softness, mode=self.mode
        )
        soil_wetness = soil_ratio.clamp(min=1e-8) ** beta
        recharge = (rainfall + to_soil) * soil_wetness
        excess = self._soft_relu_st(soil - fc, self.storage_softness)
        evap_ratio = st.clamp(
            soil / (lp * fc), 0.0, 1.0, softness=self.ratio_softness, mode=self.mode
        )
        evap_factor = (
            self._soft_relu_st(evap_ratio, softness=self.ratio_softness) ** theta['gamma']
        )
        et = self._soft_min_st(
            soil,
            Ep * evap_factor,
            self.storage_softness,
        )

        # Percolation
        perc = self._soft_min_st(
            upper_zone,
            theta['perc'],
            self.storage_softness,
        )

        # Compute the fluxes
        q0 = theta['k0'] * self._soft_relu_st(
            upper_zone - theta['uzl'], self.storage_softness
        )
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

    def _soft_relu_st(self, x: torch.Tensor, softness: float) -> torch.Tensor:
        """Hard relu forward, soft relu backwards."""
        return st.relu_st(x, softness=softness, mode=self.mode)

    def _soft_min_st(
        self, a: torch.Tensor, b: torch.Tensor, softness: float
    ) -> torch.Tensor:
        """Hard minimum on forward, soft minimum backwards."""
        return st.min_st(
            torch.stack((a, b), dim=-1),
            dim=-1,
            softness=softness,
            mode=self.mode,
        ).values
