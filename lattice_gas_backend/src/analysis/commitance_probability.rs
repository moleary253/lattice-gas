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
    use ndarray::arr1;

    #[test]
    fn basic_functionality() {
        let bottom_absorbing_size = 0;
        let top_absorbing_size = 10;

        let sizes = vec![
            1, 1, 2, 3, 4, 4, 3, 2, 1, 0, // Droplet stays same size
            1, 2, 3, 4, 5, 2, 2, 2, 1, 0, // Droplet breaks/skips size
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // Droplet hits top absorbing size
        ];

        let delta_times = Vec::from_iter((1..sizes.len()).map(|x| x as f64));

        let (succeeded, seen) = commitance(
            &sizes,
            &delta_times,
            bottom_absorbing_size,
            top_absorbing_size,
        );

        let expected_seen = arr1(&[63.0, 96.0, 47.0, 49.0, 40.0, 26.0, 27.0, 28.0, 29.0]);
        let expected_succeeded = arr1(&[21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]);

        assert_eq!(expected_succeeded, succeeded);
        assert_eq!(expected_seen, seen);
    }
}
