use pyo3::prelude::*;
use rand::prelude::*;

/// Calculates the Cumulant Generating Function (CGF).
///
/// Uses a technique which is derived from cloning algorithms to calculate
/// the CGF. The technique uses the observable $A_T = \int_0^T h(w(t)) dt$,
/// where $w(t)$ represents a path through state space, and $h(t)$ is equal
/// to $1$ when the simulation has not hit the go signal, and $0$ afterwards.
///
/// :param finishing_times: The time at which each simulation reached the "go"
/// signal
/// :param k: The argument of the cumulant generating function. Should be
/// positive. Calculates the cumulant generating function of `-k`, which
/// corresponds to the rate at which products with `k_off = k` are produced.
/// :param seed: The random seed of the simulation.
///
/// :return: $K(-k)$
#[pyfunction]
pub fn calculate_cgf(finishing_times: Vec<f64>, k: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut cloning_sim_times = Vec::with_capacity(finishing_times.len());
    let mut still_active_times = finishing_times.clone();

    still_active_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cloning_sim_times.push(still_active_times.remove(0));

    let mut current_time = cloning_sim_times[0];
    while cloning_sim_times.len() < finishing_times.len() {
        let sum_of_rates = k * cloning_sim_times.len() as f64 * still_active_times.len() as f64
            / finishing_times.len() as f64;
        let dt = -(rng.random::<f64>()).ln() / sum_of_rates;
        if current_time + dt > still_active_times[0] {
            current_time = still_active_times.remove(0);
            cloning_sim_times.push(current_time);
            continue;
        }

        let chosen_time_index =
            (rng.random::<f64>() * still_active_times.len() as f64).floor() as usize;
        current_time += dt;
        still_active_times.remove(chosen_time_index);
        cloning_sim_times.push(current_time);
    }

    cloning_sim_times
}
