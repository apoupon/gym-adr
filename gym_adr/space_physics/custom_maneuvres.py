# Custom Orbital maneuvers.


import numpy as np
from astropy import units as u

from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
from poliastro.core.elements import coe_rotation_matrix, rv2coe
from poliastro.util import norm

from gym_adr.space_physics.custom_low_level import hohmann_any_angle
from gym_adr.space_physics.in_plane_physics import delta_u


# Helper functioms
def to_mean(nu, argp):
    """
    Convert true anomaly and argument of perigee to mean anomaly.

    Parameters
    ----------
    nu : astropy.units.Quantity
        True anomaly in degrees.
    argp : astropy.units.Quantity
        Argument of perigee in degrees.

    Returns
    -------
    mean : astropy.units.Quantity
        Mean anomaly in degrees.
    """
    # What is seen on the graph
    mean = (nu + argp) << u.deg
    if mean > 180 * u.deg:
        mean = -360 * u.deg + mean  # wrap to [-180,180]
    return mean


def time_to_inc_change(orbit: Orbit):
    """
    Compute inc change thrust location and time of flight to reach inc change location.
    Must be done BEFORE inc change

    Parameters
    ----------
    orbit : Orbit
        Orbit to compute the inc change for.

    Returns
    -------
    time : astropy.units.Quantity
        Time to reach the inc change location.
    thrust_location : astropy.units.Quantity
        Location of the thrust in degrees.
    """
    # Compute location by converting nu to mean
    mean_anomaly = to_mean(orbit.nu, orbit.argp)
    thrust_location = 0 * u.deg if mean_anomaly <= 0 else 179.99999999999 * u.deg

    # Compute time
    delta_mean = thrust_location - mean_anomaly  # should always be positive
    time = (orbit.period << u.s) * delta_mean / (360 * u.deg)

    return time, thrust_location


def time_to_raan_change(orbit_i: Orbit, orbit_f: Orbit):
    """
    Compute inc change thrust location and time of flight to reach raan change location
    Must be done AFTER inc change

    Parameters
    ----------
    orbit_i : Orbit
        Initial orbit.
    orbit_f : Orbit
        Final orbit.

    Returns
    -------
    time : astropy.units.Quantity
        Time to reach the raan change location.
    u_final : astropy.units.Quantity
        Final argument of latitude after the raan change.
    theta : astropy.units.Quantity
        Angle between the initial and final RAANs.
    """
    # Compute required values for equations
    delta_raan = (orbit_f.raan - orbit_i.raan) << u.rad
    i_initial = orbit_i.inc << u.rad

    # Compute thrust location
    theta = np.arccos(
        np.cos(i_initial) ** 2 + np.cos(delta_raan) * np.sin(i_initial) ** 2
    )  # [rad]
    theta = theta * -np.sign(delta_raan)

    u_final = np.arccos(
        np.cos(i_initial)
        * np.sin(i_initial)
        * ((1 - np.cos(delta_raan)) / np.sin(theta))
    )  # [rad]

    # Compute time
    u_orbit_i = (orbit_i.nu + orbit_i.argp) << u.deg
    delta_u = (u_final - u_orbit_i) << u.deg
    if delta_u < 0:
        delta_u = 360 * u.deg + delta_u  # wrap to 360
    time = (orbit_i.period << u.s) * delta_u / (360 * u.deg)

    return time, u_final, theta


def hohmann_with_phasing(orbit_i: Orbit, orbit_f: Orbit, debug=True):
    """
    Compute a Hohmann transfer with correct phasing to a target debris.
    For circular orbits only.

    Parameters
    ----------
    orbit_i : Orbit
        Initial orbit.
    orbit_f : Orbit
        Final orbit.

    Returns
    -------
    Maneuver
        Maneuver object containing the two burns needed for the transfer.
    """
    # Downwards Hohmann
    down = False

    # Calculate transfer time for delta_u
    r_f = orbit_f.a
    r_f = r_f.to_value(u.m)
    rv = orbit_i.rv()
    rv = (rv[0].to_value(u.m), rv[-1].to_value(u.m / u.s))
    k = orbit_i.attractor.k
    _, _, t_trans = hohmann_any_angle(k, rv, r_f)

    # Calculate delta at which the burn should be applied
    target_delta = delta_u(t_trans, orbit_f)
    if target_delta < 0:
        down = True
        target_delta = 360 * u.deg + target_delta  # wrap to 360

    # Calulate the current delta
    mean_anomaly_i = (orbit_i.nu + orbit_i.argp) << u.deg
    mean_anomaly_f = (orbit_f.nu + orbit_f.argp) << u.deg
    current_delta = mean_anomaly_f - mean_anomaly_i << u.deg
    if current_delta < 0:
        current_delta = 360 * u.deg + current_delta  # wrap to 360

    # Calculate the angular velocities
    w_i = orbit_i.n.to(u.deg / u.s)
    w_f = orbit_f.n.to(u.deg / u.s)

    # Calculate the time to the first burn
    dist = current_delta - target_delta if not down else target_delta - current_delta

    if dist < 0:
        dist = 360 * u.deg + dist  # wrap to 360
    t_1 = dist / np.abs(w_i - w_f)

    # Propagate to the first burn
    orbit_i = orbit_i.propagate(t_1)
    orbit_f = orbit_f.propagate(t_1)

    if debug:
        mean_anomaly_i = (orbit_i.nu + orbit_i.argp) << u.deg
        mean_anomaly_f = (orbit_f.nu + orbit_f.argp) << u.deg

    # Compute delta_v vectors from first burn location
    r_f = orbit_f.a
    r_f = r_f.to_value(u.m)
    rv = orbit_i.rv()
    rv = (rv[0].to_value(u.m), rv[-1].to_value(u.m / u.s))

    # Calculate hohmann DV and Transfer Time from the first burn location
    k = orbit_i.attractor.k
    dv_a, dv_b, t_trans = hohmann_any_angle(k, rv, r_f)
    dv_a, dv_b, t_trans = dv_a * u.m / u.s, dv_b * u.m / u.s, t_trans * u.s
    t_2 = t_trans

    return Maneuver(
        (t_1.decompose(), dv_a.decompose()),
        (t_2.decompose(), dv_b.decompose()),
    )


def simple_inc_change(orbit_i: Orbit, orbit_f: Orbit):
    """
    Compute thrust vectors and phase time needed for an inclination change.

    Parameters
    ----------
    orbit_i : Orbit
        Initial orbit.
    orbit_f : Orbit
        Final orbit.

    Returns
    -------
    Maneuver
        Maneuver object containing the thrust vector and time to apply it.
    """
    # Compute thrust location
    time_to_thrust, thrust_location = time_to_inc_change(orbit_i)
    orbit_i = orbit_i.propagate(time_to_thrust)

    # Calculate the thrust value
    v = norm(orbit_i.v << u.m / u.s)
    inc_i = orbit_i.inc << u.rad
    inc_f = orbit_f.inc << u.rad
    inc_delta = inc_f - inc_i
    thrust_norm = 2 * v * np.sin((inc_delta << u.rad) / 2)

    # Calculate the thrust vector
    y_thrust = np.sin(inc_delta / 2) * thrust_norm
    z_thrust = -np.cos(inc_delta / 2) * thrust_norm

    # Rotate values through nu
    # thrust_vector = np.array([0 ,y_thrust.value,z_thrust.value]) * u.m / u.s
    nu = orbit_i.nu << u.rad

    thrust_vector = (
        np.array(
            [-np.sin(nu) * y_thrust.value, np.cos(nu) * y_thrust.value, z_thrust.value]
        )
        * u.m
        / u.s
    )

    if thrust_location == 0 * u.deg:
        thrust_vector = -thrust_vector
    else:
        thrust_vector[1] = -thrust_vector[1]

    # Use rotation matrix to go from orbital to general referential
    k = orbit_i.attractor.k
    rv = orbit_i.rv()
    rv = (rv[0].to_value(u.m), rv[-1].to_value(u.m / u.s))
    _, ecc, inc, raan, argp, nu = rv2coe(k, *rv)
    rot_matrix = coe_rotation_matrix(inc, raan, argp)
    thrust_vector = rot_matrix @ thrust_vector

    return Maneuver((time_to_thrust.decompose(), thrust_vector.decompose()))


def simple_raan_change(orbit_i: Orbit, orbit_f: Orbit):
    """
    Compute thrust vectors and phase time needed for an inclination change.

    Parameters
    ----------
    orbit_i : Orbit
        Initial orbit.
    orbit_f : Orbit
        Final orbit.

    Returns
    -------
    Maneuver
        Maneuver object containing the thrust vector and time to apply it.
    """
    # Compute thrust location and theta
    time_to_thrust, thrust_location, theta = time_to_raan_change(orbit_i, orbit_f)
    orbit_i = orbit_i.propagate(time_to_thrust)

    # Calculate the thrust value
    v = norm(orbit_i.v << u.m / u.s)
    # inc_i = orbit_i.inc << u.rad
    # inc_f = orbit_f.inc << u.rad
    inc_delta = -theta
    thrust_norm = 2 * v * np.sin((inc_delta << u.rad) / 2)

    # Calculate the thrust vector
    y_thrust = np.sin(inc_delta / 2) * thrust_norm
    z_thrust = -np.cos(inc_delta / 2) * thrust_norm

    # Rotate values through nu
    # thrust_vector = np.array([0 ,y_thrust.value,z_thrust.value]) * u.m / u.s
    nu = orbit_i.nu << u.rad
    thrust_vector = (
        np.array(
            [-np.sin(nu) * y_thrust.value, np.cos(nu) * y_thrust.value, z_thrust.value]
        )
        * u.m
        / u.s
    )
    thrust_vector = -thrust_vector

    # Use rotation matrix to go from orbital to general referential
    k = orbit_i.attractor.k
    rv = orbit_i.rv()
    rv = (rv[0].to_value(u.m), rv[-1].to_value(u.m / u.s))
    _, ecc, inc, raan, argp, nu = rv2coe(k, *rv)
    rot_matrix = coe_rotation_matrix(inc, raan, argp)
    thrust_vector = rot_matrix @ thrust_vector

    return Maneuver((time_to_thrust.decompose(), thrust_vector.decompose()))
