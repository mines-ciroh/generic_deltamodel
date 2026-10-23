import softtorch as st
import torch
from hydrodl2.core.calc import uh_conv, uh_gamma

from .base_state_equations import BaseStateEquations


class HbvStateEquations(BaseStateEquations):
    """State equations for the HBV hydrological model.

    Adapted from `HBV.forward` in hydrodl2's `hbv_adj.py` (Song et al., 2024).
    """

    n_stores = 5

    # Routing parameters
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
        len_uh: int = 15,  # This was pulled directly from hbv_adj.py.
    ) -> torch.Tensor:
        """Route streamflow [time, basins] with a gamma unit hydrograph.

        `rout_a` and `rout_b` are routing parameters of shape [basins].
        """
        # TODO: revisit this.
        n_steps = streamflow.shape[0]

        # the two routing parameters
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
        snowpack = st.relu(state[0])
        meltwater = st.relu(state[1])
        soil = st.relu(state[2])
        upper_zone = st.relu(state[3])
        lower_zone = st.relu(state[4])

        # Snow module
        # ---
        # Grab the fraction of precipitation as snow vs. rain with st.less()
        snow_fraction = st.less(T, tt)
        rain_fraction = 1 - snow_fraction
        snowfall = P * snow_fraction
        rainfall = P * rain_fraction

        refreeze = st.min(torch.stack((cfr * cfmax * st.relu(tt - T), meltwater), dim=-1))

        melt = st.min(torch.stack((cfmax * st.relu(T - tt), snowpack), dim=-1))

        to_soil = st.relu(
            meltwater - cwh * snowpack,
        )

        # Soil moisture and evapotranspiration (ET) module
        # ---
        soil_ratio = st.clamp(
            soil / fc,
            0.0,  # lower
            1.0,  # upper
        )
        soil_wetness = st.clamp(soil_ratio**beta, 0.0, 1.0)
        recharge = (rainfall + to_soil) * soil_wetness
        excess = st.relu(soil - fc)
        evap_ratio = st.clamp(soil / (lp * fc), 0.0, 1.0)
        evap_factor = st.relu(evap_ratio) ** theta['gamma']
        et = st.min(torch.stack((soil, Ep * evap_factor), dim=-1))

        # Percolation
        # ---
        perc = st.min(torch.stack((upper_zone, theta['perc']), dim=-1))

        # Compute the three fluxes
        # ---
        q0 = theta['k0'] * st.relu(
            upper_zone - theta['uzl'],
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
