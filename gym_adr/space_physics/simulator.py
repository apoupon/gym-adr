import copy
import numpy as np
import scipy.io
import pandas as pd

from astropy import units as u
from poliastro.bodies import Earth

from gym_adr.space_physics.custom_orbit import Orbit
import gym_adr.space_physics.custom_maneuvres as custom_maneuvres


class Debris:
    """
    Object for representing a single piece of orbital debris.

    Parameters
    ----------
    poliastro_orbit : poliastro.twobody.orbit.Orbit
        Orbit of the debris object.
    norad_id : int
        Unique NORAD ID associated with the debris.
    """

    def __init__(self, poliastro_orbit, norad_id):
        self.poliastro_orbit = poliastro_orbit
        self.norad_id = norad_id


class Simulator:
    """
    Simulation environment for orbital debris interception using a maneuverable OTV (orbital transfer vehicle).

    Parameters
    ----------
    starting_index : int, optional
        Index of the debris object to initialize the OTV with. Default is 0.
    n_debris : int, optional
        Number of debris objects to load into the simulation. Default is 10.
    starting_fuel : float, optional
        Initial fuel available to the OTV. Default is 1000.0.
    """

    def __init__(
        self,
        starting_index: int = 0,
        n_debris: int = 10,
        starting_fuel: float = 1000.0,
    ):
        # Initialise the debris dictionary and assign the otv to an Orbit
        # self.debris_list = self.init_random_debris(n=n_debris)
        self.debris_list = self.debris_from_dataset(
            n=n_debris
        )  # dataset contains 320 debris

        self.otv_orbit = copy.copy(self.debris_list[starting_index].poliastro_orbit)
        self.current_fuel = starting_fuel

    def simulate_action(self, action, render=False, step_sec=15):
        """
        Simulate the execution of a transfer maneuver based on a given action.

        Parameters
        ----------
        action : int
            Index of the target debris (used to extract NORAD ID).
        render : bool, optional
            If True, returns a detailed dataframe of orbital states during transfer. Default is False.
        step_sec : int, optional
            Time step in seconds for maneuver rendering. Default is 15.

        Returns
        -------
        DV_required : astropy.units.Quantity
            Total delta-v required to complete the transfer.
        DT_required : astropy.units.Quantity
            Total time of flight required for the transfer.
        or
        location_frames_df : pandas.DataFrame
            Time series of orbital positions and states (if render=True).
        """
        location_frames_df, DV_required, DT_required = self.strategy_1(
            action, render=render, step_sec=step_sec
        )

        if render:
            return location_frames_df
        return DV_required, DT_required

    def strategy_1(self, action, render=False, step_sec=15):
        """
        Execute a 3-phase orbital transfer strategy:
        1. Inclination change
        2. RAAN alignment
        3. Hohmann transfer with phasing

        Parameters
        ----------
        action : int
            Index of the target debris in the debris list.
        render : bool, optional
            If True, collect orbital states into a dataframe for visualization. Default is False.
        step_sec : int, optional
            Time step in seconds for simulation during rendering. Default is 15.

        Returns
        -------
        location_frames_df : pandas.DataFrame or None
            DataFrame containing orbital states at each time step (if render=True).
        total_dv : astropy.units.Quantity
            Total delta-v cost for the complete maneuver.
        min_time : astropy.units.Quantity
            Total time of flight required to complete all phases.
        """
        # Set the target from the action
        target_debris = self.debris_list[action].poliastro_orbit

        # ---- Inclination change
        inc_change = custom_maneuvres.simple_inc_change(self.otv_orbit, target_debris)

        # Get the transfer time of the hoh_phas
        transfer_time = inc_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, inc_frames = self.otv_orbit.apply_maneuver_custom(
            inc_change,
            copy.deepcopy(self.debris_list) if render else None,
            step_sec=step_sec,
            render=render,
        )
        # Append the current fuel to the frames df
        if render:
            inc_frames["fuel"] = self.current_fuel
        self.current_fuel -= inc_change.get_total_cost().value

        # Propagate all debris to the end of the transfer
        for i, debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(
                transfer_time
            )

        # ---- Raan change
        target_debris = self.debris_list[action].poliastro_orbit
        raan_change = custom_maneuvres.simple_raan_change(self.otv_orbit, target_debris)

        # Get the transfer time of the hoh_phas
        transfer_time = raan_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, raan_frames = self.otv_orbit.apply_maneuver_custom(
            raan_change,
            copy.deepcopy(self.debris_list) if render else None,
            step_sec=step_sec,
            render=render,
        )
        # Append the current fuel to the frames df
        if render:
            raan_frames["fuel"] = self.current_fuel
        self.current_fuel -= inc_change.get_total_cost().value

        # Propagate all debris to the end of the transfer
        for i, debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(
                transfer_time
            )

        # ---- Hohmann
        target_debris = self.debris_list[action].poliastro_orbit
        hoh_change = custom_maneuvres.hohmann_with_phasing(
            self.otv_orbit, target_debris
        )

        # Get the transfer time of the hoh_phas
        transfer_time = hoh_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, hoh_frames = self.otv_orbit.apply_maneuver_custom(
            hoh_change,
            copy.deepcopy(self.debris_list) if render else None,
            step_sec=step_sec,
            render=render,
        )
        # Append the current fuel to the frames df
        if render:
            hoh_frames["fuel"] = self.current_fuel
        self.current_fuel -= inc_change.get_total_cost().value

        # Propagate all debris to the end of the transfer
        for i, debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(
                transfer_time
            )

        # Total resources used
        total_dv = (
            hoh_change.get_total_cost()
            + raan_change.get_total_cost()
            + inc_change.get_total_cost()
        )
        min_time = (
            hoh_change.get_total_time()
            + raan_change.get_total_time()
            + inc_change.get_total_time()
        )

        if render:
            # Concat the dataframes
            location_frames_df = pd.concat(
                [inc_frames, raan_frames, hoh_frames], axis=0
            )
            # Add a column for the action
            location_frames_df["target_index"] = action

            return location_frames_df, total_dv, min_time

        return None, total_dv, min_time

    def init_random_debris(self, n):
        """
        Generate n synthetic debris orbits with random orbital parameters.

        Parameters
        ----------
        n : int
            Number of debris objects to generate.

        Returns
        -------
        debris_list : list of Debris
            List of Debris objects with random orbital elements.
        """
        np.random.seed(42)

        debris_list = []

        for norad_id in range(n):
            min_a = 6371 + 200
            max_a = 6371 + 10000
            a = np.random.uniform(min_a, max_a) * u.km
            ecc = 0 * u.one
            inc = np.random.uniform(0, 45) * u.deg
            raan = np.random.uniform(0, 45) * u.deg
            argp = 0 * u.deg
            nu = np.random.uniform(-180, 180) * u.deg

            debris = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
            debris_list.append(Debris(poliastro_orbit=debris, norad_id=norad_id))

        return debris_list

    def debris_from_dataset(self, n):
        """
        Load debris orbits from a TLE dataset and convert to Debris objects.

        Parameters
        ----------
        n : int
            Number of debris objects to extract from the dataset.

        Returns
        -------
        debris_list : list of Debris
            List of Debris objects loaded from the dataset.
        """
        debris_list = []
        dataset = scipy.io.loadmat("data/TLE_iridium.mat")["TLE_iridium"]

        # Select only favourable debris for rendering
        i = 0
        while len(debris_list) < n + 1:
            norad_id = dataset[0][i]
            a = dataset[6][i] * u.km
            ecc = 0 * u.one  # ecc = dataset[3][i] * u.one for more accurate modelling
            inc = dataset[1][i] * u.deg
            raan = dataset[2][i] * u.deg
            argp = 0 * u.deg  # dataset[4][i] * u.deg for more accurate modelling
            nu = (dataset[5][i] - 180) * u.deg

            if dataset[2][i] < 20:
                debris = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
                debris_list.append(Debris(poliastro_orbit=debris, norad_id=norad_id))

            i += 1

        return debris_list
