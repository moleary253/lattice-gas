use super::*;

/// Finds the largest size droplet at each time in the simulation.
///
/// Requires Droplets analyzer to also be in the simulation before this.
///
/// Attributes:
///  - sizes: A Vec of the sizes reached during the simulation.
///  - delta_times: A Vec of the times between each size change.
///  - bottom_threshold: Sizes smaller than this will be treated as 0.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct LargestDropletSizeAnalyzer {
    pub sizes: Vec<usize>,
    pub delta_times: Vec<f64>,
    #[pyo3(get)]
    pub bottom_threshold: usize,
}

#[pymethods]
impl LargestDropletSizeAnalyzer {
    #[new]
    pub fn new(bottom_threshold: usize) -> Self {
        LargestDropletSizeAnalyzer {
            sizes: Vec::new(),
            delta_times: Vec::new(),
            bottom_threshold,
        }
    }

    #[getter]
    pub fn sizes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_vec(py, self.sizes.clone())
    }

    #[getter]
    pub fn delta_times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.delta_times.clone())
    }
}

impl LargestDropletSizeAnalyzer {
    fn droplets<'a>(other_analyzers: &'a Vec<Box<dyn Analyzer>>) -> &'a Droplets {
        for analyzer in other_analyzers.iter() {
            let analyzer_any = &**analyzer as &dyn Any;
            let Some(droplets) = analyzer_any.downcast_ref::<Droplets>() else {
                continue;
            };
            return droplets;
        }
        panic!("`Droplets` was not included as an additional analyzer!");
    }

    pub fn init<'a>(&mut self, other_analyzers: &'a Vec<Box<dyn Analyzer>>) {
        self.sizes.clear();
        self.delta_times.clear();
        self.delta_times.push(0.);
        let droplets = LargestDropletSizeAnalyzer::droplets(other_analyzers);
        let mut size = droplets
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len()));
        if size < self.bottom_threshold {
            size = 0;
        }
        self.sizes.push(size);
    }

    pub fn update<'a>(&mut self, dt: f64, other_analyzers: &'a Vec<Box<dyn Analyzer>>) {
        let droplets = LargestDropletSizeAnalyzer::droplets(other_analyzers);

        *self.delta_times.last_mut().unwrap() += dt;

        let mut size = droplets
            .droplets
            .iter()
            .fold(0, |max, droplet| max.max(droplet.len()));
        if size < self.bottom_threshold {
            size = 0;
        }
        if size != *self.sizes.last().unwrap() {
            self.sizes.push(size);
            self.delta_times.push(0.);
        }
    }
}

#[typetag::serde]
impl Analyzer for LargestDropletSizeAnalyzer {
    fn init(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        _chain: &Box<dyn MarkovChain>,
        previous_analyzers: &Vec<Box<dyn Analyzer>>,
    ) {
        self.init(previous_analyzers);
    }

    fn update(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        _chain: &Box<dyn MarkovChain>,
        _reaction: BasicReaction<u32>,
        dt: f64,
        previous_analyzers: &Vec<Box<dyn Analyzer>>,
    ) {
        self.update(dt, previous_analyzers);
    }
}

#[cfg(test)]
mod tests {
    use crate::boundary_condition;

    use super::*;
    use ndarray::arr2;

    #[test]
    fn largest_droplet_size_over_time_works() {
        use crate::reaction::BasicReaction as BR;
        let mut state = arr2(&[
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let boundary = Box::new(boundary) as Box<dyn BoundaryCondition>;
        let reactions = vec![
            BR::point_change(0_u32, 1, [5, 1]),
            BR::point_change(1, 0, [5, 2]),
            BR::point_change(1, 0, [5, 1]),
            BR::point_change(0, 1, [0, 2]),
            BR::point_change(0, 1, [2, 0]),
            BR::point_change(1, 0, [3, 4]),
            BR::point_change(0, 1, [0, 1]),
            BR::point_change(1, 0, [1, 1]),
        ];
        let droplets = Droplets::new(&state.view(), &boundary, &vec![1]);
        let mut analyzers = vec![Box::new(droplets) as Box<dyn Analyzer>];
        let mut largest_size = LargestDropletSizeAnalyzer::new(0);

        largest_size.init(&analyzers);

        for reaction in reactions {
            reaction.apply(&mut state);
            let Some(droplets_mut) =
                (&mut *analyzers[0] as &mut dyn Any).downcast_mut::<Droplets>()
            else {
                panic!("Cast didn't work");
            };
            droplets_mut.update(&state.view(), &boundary, &reaction);

            largest_size.update(1.0, &analyzers);
        }

        let expected_sizes = vec![6, 7, 13, 10, 11, 10];
        let expected_dts = vec![4.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(expected_sizes, largest_size.sizes);
        assert_eq!(expected_dts, largest_size.delta_times);
    }

    #[test]
    fn bottom_threshold() {
        use crate::reaction::BasicReaction as BR;
        let mut state = arr2(&[
            [1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0],
        ]);
        let boundary = boundary_condition::Periodic;
        let boundary = Box::new(boundary) as Box<dyn BoundaryCondition>;
        let reactions = vec![
            BR::point_change(0_u32, 1, [5, 1]),
            BR::point_change(1, 0, [5, 2]),
            BR::point_change(1, 0, [5, 1]),
            BR::point_change(0, 1, [0, 2]),
            BR::point_change(0, 1, [2, 0]),
            BR::point_change(1, 0, [3, 4]),
            BR::point_change(0, 1, [0, 1]),
            BR::point_change(1, 0, [1, 1]),
        ];
        let droplets = Droplets::new(&state.view(), &boundary, &vec![1]);
        let mut analyzers = vec![Box::new(droplets) as Box<dyn Analyzer>];
        let mut largest_size = LargestDropletSizeAnalyzer::new(11);

        largest_size.init(&analyzers);

        for reaction in reactions {
            reaction.apply(&mut state);
            let Some(droplets_mut) =
                (&mut *analyzers[0] as &mut dyn Any).downcast_mut::<Droplets>()
            else {
                panic!("Cast didn't work");
            };
            droplets_mut.update(&state.view(), &boundary, &reaction);

            largest_size.update(1.0, &analyzers);
        }

        let expected_sizes = vec![0, 13, 0, 11, 0];
        let expected_dts = vec![5.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(expected_sizes, largest_size.sizes);
        assert_eq!(expected_dts, largest_size.delta_times);
    }
}
