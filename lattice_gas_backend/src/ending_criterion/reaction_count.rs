use super::*;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[pyclass]
pub struct ReactionCount {
    pub threshold: usize,
    count: usize,
}

#[pymethods]
impl ReactionCount {
    #[new]
    pub fn new(threshold: usize) -> Self {
        ReactionCount {
            threshold,
            count: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

impl<T: Clone, R: Reaction<T>> EndingCriterion<T, R> for ReactionCount {
    fn should_end(&self) -> bool {
        self.count >= self.threshold
    }

    fn initialize(
        &mut self,
        _system: &ArrayView2<T>,
        _chain: &Box<dyn MarkovChain<T, R>>,
        _boundary: &Box<dyn BoundaryCondition<T>>,
    ) {
        self.count = 0;
    }

    fn update(
        &mut self,
        _system: &ArrayView2<T>,
        _chain: &Box<dyn MarkovChain<T, R>>,
        _boundary: &Box<dyn BoundaryCondition<T>>,
        _delta_t: f64,
        _reaction: R,
    ) {
        self.count += 1;
    }
}
