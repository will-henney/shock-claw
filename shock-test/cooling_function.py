DEFAULTS = {
    "cool_method": "second order",
    "T_eq": 1.0,
    "cool_rate": 1.0,
    # Cooling increases steeply with temperature
    "cool_slope": 2.3,
    # Heating decreases gradually with temperature
    "heat_slope": -0.5,
    # Monatomic ideal gas
    "gamma": 5/3,
    "gamma1": 2/3,
}



def cooling_source_term_step(solver, state, dt):
    """Apply radiative heating and cooling over timestep

    For use with classic-type Clawpack solvers, for example:
    
          solver.step_source = cooling_source_term_step

    Heating and cooling are both proportional to density squared, and
    are power-law functions of temperature, with power-law indices
    given by the `cool_slope` and `heat_slope` parameters (all
    parameters can be set via the `state.problem_data` dictionary).
    The equilibrium temperature (where heating = cooling) is given by
    the `T_eq` parameter.  The cooling rate (energy/time) for
    density=1 and temperature=T_eq is given by the `cool_rate`
    parameter.  The cooling method can be either "first order" or
    "second order".,

    Author: William Henney, 2024
    """
    # Attempt to get the cooling parameters from the state, with
    # fallback to the defaults
    cool_method = state.problem_data.get("cool_method", DEFAULTS["cool_method"])
    T_eq = state.problem_data.get("T_eq", DEFAULTS["T_eq"])
    cool_rate = state.problem_data.get("cool_rate", DEFAULTS["cool_rate"])
    cool_slope = state.problem_data.get("cool_slope", DEFAULTS["cool_slope"])
    heat_slope = state.problem_data.get("heat_slope", DEFAULTS["heat_slope"])
    gamma = state.problem_data.get("gamma", DEFAULTS["gamma"])
    gamma1 = state.problem_data.get("gamma1", DEFAULTS["gamma1"])
    
    # Find primitive variables from conserved variables
    density = state.q[0, :]
    velocity = state.q[1, :] / density
    internal_energy = state.q[2, :] - 0.5 * density * velocity**2
    pressure = gamma1 * internal_energy
    temperature = pressure / density

    # Integrate net (heating - cooling) over the timestep
    if cool_method == "first order":
        # Calculate change of energy over timestep (positive for net
        # heating)
        dE = dt * cool_rate * density**2 * (
            (temperature / T_eq) ** heat_slope
            - (temperature / T_eq) ** cool_slope
        )
    elif cool_method == "second order":
        # Take half timestep at constant density
        dE2 = 0.5 * dt * cool_rate * density**2 * (
             (temperature / T_eq) ** heat_slope
            - (temperature / T_eq) ** cool_slope
        )
        # Recalculate temperature after half timestep
        internal_energy += dE2
        temperature = pressure / density
        # Then take the full timestep
        dE = dt * cool_rate * density**2 * (
            (temperature / T_eq) ** heat_slope
            - (temperature / T_eq) ** cool_slope
        )
    elif cool_method == "exact":
        raise NotImplementedError("Exact cooling is not implemented yet")
    else:
        raise ValueError(f"Unknown cooling method: {cool_method}")
    
    # Modify the energy of the state
    state.q[2, :] += dE

    return None
