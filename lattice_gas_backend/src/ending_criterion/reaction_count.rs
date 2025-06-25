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

#[typetag::serde]
impl EndingCriterion for ReactionCount {
    fn should_end(&self) -> bool {
        self.count >= self.threshold
    }

    fn initialize(
        &mut self,
        _system: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
    ) {
        self.count = 0;
    }

    fn update(
        &mut self,
        _system: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
        _delta_t: f64,
        _reaction: BasicReaction<u32>,
    ) {
        self.count += 1;
    }
}
