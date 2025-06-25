use super::*;

#[pyfunction]
#[pyo3(name = "commitance")]
pub fn py_commitance<'py>(
    py: Python<'py>,
    sizes: Vec<usize>,
    delta_times: Vec<f64>,
    bottom_absorbing_size: usize,
    top_absorbing_size: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mut time_seen = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);
    let mut time_succeeded = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);

    let mut tracking_start = None;
    for (i, &size) in sizes.iter().enumerate() {
        advance_one_step(
            &sizes,
            &mut tracking_start,
            bottom_absorbing_size,
            top_absorbing_size,
            &mut time_seen,
            &mut time_succeeded,
            &delta_times,
            i,
            size,
        );
        py.check_signals()?;
    }
    Ok((time_succeeded.to_pyarray(py), time_seen.to_pyarray(py)))
}

pub fn commitance(
    sizes: &Vec<usize>,
    delta_times: &Vec<f64>,
    bottom_absorbing_size: usize,
    top_absorbing_size: usize,
) -> (Array1<f64>, Array1<f64>) {
    let mut time_seen = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);
    let mut time_succeeded = Array1::zeros(top_absorbing_size - 1 - bottom_absorbing_size);

    let mut tracking_start = None;
    for (i, &size) in sizes.iter().enumerate() {
        advance_one_step(
            &sizes,
            &mut tracking_start,
            bottom_absorbing_size,
            top_absorbing_size,
            &mut time_seen,
            &mut time_succeeded,
            delta_times,
            i,
            size,
        );
    }

    (time_succeeded, time_seen)
}

fn advance_one_step(
    sizes: &Vec<usize>,
    tracking_start: &mut Option<usize>,
    bottom_absorbing_size: usize,
    top_absorbing_size: usize,
    time_seen: &mut Array1<f64>,
    time_succeeded: &mut Array1<f64>,
    delta_times: &Vec<f64>,
    i: usize,
    size: usize,
) {
    if *tracking_start == None {
        if size > bottom_absorbing_size {
            *tracking_start = Some(i);
        }
        return;
    }
    if size >= top_absorbing_size {
        for j in tracking_start.map(std::convert::identity).unwrap()..i {
            time_seen[sizes[j] - bottom_absorbing_size - 1] += delta_times[j];
            time_succeeded[sizes[j] - bottom_absorbing_size - 1] += delta_times[j];
        }
        *tracking_start = None;
        return;
    }
    if size <= bottom_absorbing_size {
        for j in tracking_start.map(std::convert::identity).unwrap()..i {
            time_seen[sizes[j] - bottom_absorbing_size - 1] += delta_times[j];
        }
        *tracking_start = None;
        return;
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
        let initial_state = arr2(&[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]);
        let boundary = crate::boundary_condition::Periodic;
        let counts_as_droplet = vec![1];
        let bottom_absorbing_size = 0;
        let top_absorbing_size = 10;

        let reactions = vec![
            // Droplet size stays same as other reaction happens
            BR::point_change(0_u32, 1, [1, 1]),
            BR::point_change(0, 1, [6, 6]),
            BR::point_change(0, 1, [1, 2]),
            BR::point_change(0, 1, [2, 2]),
            BR::point_change(0, 1, [2, 1]),
            BR::point_change(1, 0, [6, 6]),
            BR::point_change(1, 0, [1, 2]),
            BR::point_change(1, 0, [2, 2]),
            BR::point_change(1, 0, [2, 1]),
            BR::point_change(1, 0, [1, 1]),
            // Droplet breaks
            BR::point_change(0, 1, [2, 1]),
            BR::point_change(0, 1, [2, 2]),
            BR::point_change(0, 1, [2, 3]),
            BR::point_change(0, 1, [2, 4]),
            BR::point_change(0, 1, [2, 5]),
            BR::point_change(1, 0, [2, 3]),
            BR::point_change(1, 0, [2, 1]),
            BR::point_change(1, 0, [2, 2]),
            BR::point_change(1, 0, [2, 4]),
            BR::point_change(1, 0, [2, 5]),
            // Droplet hits top_absorbing_size
            BR::point_change(0, 1, [2, 1]),
            BR::point_change(0, 1, [2, 2]),
            BR::point_change(0, 1, [3, 1]),
            BR::point_change(0, 1, [3, 2]),
            BR::point_change(0, 1, [4, 1]),
            BR::point_change(0, 1, [4, 2]),
            BR::point_change(0, 1, [5, 1]),
            BR::point_change(0, 1, [5, 2]),
            BR::point_change(0, 1, [6, 1]),
            BR::point_change(0, 1, [6, 2]),
        ];
        let reactions: Vec<Box<dyn Reaction<u32>>> = reactions
            .iter()
            .map(|reaction| Box::new(*reaction) as Box<dyn Reaction<u32>>)
            .collect();

        let times = vec![1.0; reactions.len()];
        let sizes = largest_droplet_size_over_time(
            &initial_state.view(),
            &(Box::new(boundary) as Box<dyn BoundaryCondition>),
            &reactions,
            counts_as_droplet,
        );

        let (succeeded, seen) =
            commitance(&sizes, &times, bottom_absorbing_size, top_absorbing_size);

        let expected_seen = arr1(&[6.0, 7.0, 4.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0]);
        let expected_succeeded = arr1(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        assert_eq!(expected_succeeded, succeeded);
        assert_eq!(expected_seen, seen);
    }
}
