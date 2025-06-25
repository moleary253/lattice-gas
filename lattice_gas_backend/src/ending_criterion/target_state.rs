use super::*;
use numpy::ndarray::Array2;
use numpy::prelude::*;
use numpy::PyArray2;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[pyclass]
pub struct TargetState {
    pub state: Array2<u32>,
    is_done: bool,
}

#[pymethods]
impl TargetState {
    #[new]
    pub fn new(state: &Bound<'_, PyArray2<u32>>) -> Self {
        TargetState {
            state: state.to_owned_array(),
            is_done: false,
        }
    }
}

#[typetag::serde]
impl EndingCriterion for TargetState {
    fn should_end(&self) -> bool {
        self.is_done
    }

    fn initialize(
        &mut self,
        _system: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
    ) {
        self.is_done = false;
    }

    fn update(
        &mut self,
        state: &ArrayView2<u32>,
        _chain: &Box<dyn MarkovChain>,
        _boundary: &Box<dyn BoundaryCondition>,
        _delta_t: f64,
        _reaction: BasicReaction<u32>,
    ) {
        self.is_done = state == self.state;
    }
}
