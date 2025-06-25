use super::*;

/// Infers the forward and backwards rates from CNT data.
///
/// More specifically, finds the amount of time spent at each size, and the
/// number of forwards and backwards events at those sizes during the
/// simulation. The rates can be straightforwardly calculated from these
/// values, and this allows combining results from multiple simulations via
/// simple addition.
///
/// Arguments:
///  - sizes: The largest size at each step
///  - delta_times: The time between each reaction
///
/// Returns `(time_spent, num_fwd, num_bkwd)`. The forwards rates are
/// `num_fwd / time_spent[:-1]`, and the backwards rates are
/// `num_bkwd / time_spent[1:]`.
#[pyfunction]
#[pyo3(name = "cnt_rates")]
pub fn py_cnt_rates<'py>(
    py: Python<'py>,
    sizes: Vec<usize>,
    delta_times: Vec<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
)> {
    let max_size = sizes.iter().max().unwrap();

    let mut time_seen = Array1::zeros(max_size + 1);
    let mut num_fwd = Array1::zeros(time_seen.len() - 1);
    let mut num_bkwd = Array1::zeros(time_seen.len() - 1);

    let mut prev_size = sizes[0];
    for (i, &size) in sizes.iter().skip(1).enumerate() {
        advance_one_step(
            &mut time_seen,
            &mut num_fwd,
            &mut num_bkwd,
            &mut prev_size,
            delta_times[i],
            size,
        );
        py.check_signals()?;
    }

    Ok((
        time_seen.to_pyarray(py),
        num_fwd.to_pyarray(py),
        num_bkwd.to_pyarray(py),
    ))
}

/// Infers the forward and backwards rates from CNT data.
///
/// More specifically, finds the amount of time spent at each size, and the
/// number of forwards and backwards events at those sizes during the
/// simulation. The rates can be straightforwardly calculated from these
/// values, and this allows combining results from multiple simulations via
/// simple addition.
///
/// Arguments:
///  - sizes: The largest size at each step
///  - delta_times: The time between each reaction
///
/// Returns `(time_spent, num_fwd, num_bkwd)`. The forwards rates are
/// `num_fwd / time_spent[:-1]`, and the backwards rates are
/// `num_bkwd / time_spent[1:]`.
pub fn cnt_rates(
    sizes: &Vec<usize>,
    delta_times: &Vec<f64>,
) -> (Array1<f64>, Array1<usize>, Array1<usize>) {
    let max_size = sizes.iter().max().unwrap();

    let mut time_seen = Array1::zeros(max_size + 1);
    let mut num_fwd = Array1::zeros(time_seen.len() - 1);
    let mut num_bkwd = Array1::zeros(time_seen.len() - 1);

    let mut prev_size = sizes[0];
    for (i, &size) in sizes.iter().skip(1).enumerate() {
        advance_one_step(
            &mut time_seen,
            &mut num_fwd,
            &mut num_bkwd,
            &mut prev_size,
            delta_times[i],
            size,
        );
    }

    (time_seen, num_fwd, num_bkwd)
}

fn advance_one_step(
    time_seen: &mut Array1<f64>,
    num_fwd: &mut Array1<usize>,
    num_bkwd: &mut Array1<usize>,
    prev_size: &mut usize,
    dt: f64,
    size: usize,
) {
    time_seen[*prev_size] += dt;

    if *prev_size < size {
        num_fwd[*prev_size] += 1;
        *prev_size = size;
    } else if *prev_size > size {
        num_bkwd[*prev_size - 1] += 1;
        *prev_size = size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary_condition::BoundaryCondition;
    use crate::reaction::BasicReaction as BR;
    use crate::reaction::Reaction;
    use ndarray::{arr1, arr2};

    #[test]
    fn basic_functionality() {
        let initial_state = arr2(&[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        let boundary = crate::boundary_condition::Periodic;
        let counts_as_droplet = vec![1];

        let reactions: Vec<BR<u32>> = vec![
            BR::point_change(0, 1, [1, 1]),
            BR::point_change(0, 1, [1, 2]),
            BR::point_change(0, 1, [3, 3]),
            BR::point_change(1, 0, [1, 1]),
            BR::point_change(1, 0, [3, 3]),
            BR::point_change(0, 1, [1, 1]),
            BR::point_change(1, 0, [1, 2]),
            BR::point_change(1, 0, [1, 1]),
        ];
        let reactions: Vec<Box<dyn Reaction<u32>>> = reactions
            .iter()
            .map(|reaction| Box::new(*reaction) as Box<dyn Reaction<u32>>)
            .collect();

        let times = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
        let sizes = largest_droplet_size_over_time(
            &initial_state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition>),
            &reactions,
            counts_as_droplet.clone(),
        );

        let (time_seen, num_fwd, num_bkwd) = cnt_rates(&sizes, &times);

        let expected_seen = arr1(&[1.0, 178.0, 76.0]);
        let expected_fwd = arr1(&[1, 2]);
        let expected_bkwd = arr1(&[1, 2]);

        assert_eq!(expected_seen, time_seen);
        assert_eq!(expected_fwd, num_fwd);
        assert_eq!(expected_bkwd, num_bkwd);
    }
}
