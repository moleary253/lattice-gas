use super::*;
use crate::reaction::BasicReaction;
use serde::{Deserialize, Serialize};

/// Describes a chain which is meant to represent a ladder-like network which
/// follows CNT rates. It assumes a 2D droplet where the free energy difference
/// comes from a thermodynamic driving force from supersaturation and a surface
/// tension term.
///
/// Since it is meant for a ladder like network, it assumes the state is of
/// shape (2, l) for some l. The state should always have a 1 at some point and
/// be zeros elsewhere. The 0th row is the 'top' process and the 1st row is the
/// 'bot' process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct CNTLadderChain {
    pub beta: f64,
    pub top_driving_force: f64,
    pub top_surface_tension: f64,
    pub bot_driving_force: f64,
    pub bot_surface_tension: f64,
    pub rescue_rate: f64,
    pub off_rate: f64,
}

#[pymethods]
impl CNTLadderChain {
    #[new]
    pub fn new(
        beta: f64,
        top_driving_force: f64,
        top_surface_tension: f64,
        bot_driving_force: f64,
        bot_surface_tension: f64,
        rescue_rate: f64,
        off_rate: f64,
    ) -> Self {
        CNTLadderChain {
            beta,
            top_driving_force,
            top_surface_tension,
            bot_driving_force,
            bot_surface_tension,
            rescue_rate,
            off_rate,
        }
    }
}

impl CNTLadderChain {
    pub fn is_possible(&self, state: &ArrayView2<u32>, reaction: BasicReaction<u32>) -> bool {
        match reaction {
            BasicReaction::Diffusion { from, to: _to } => state[from] == 1,
            _ => false,
        }
    }

    pub fn free_energy(&self, pos: [usize; 2]) -> f64 {
        if pos[0] == 0 {
            return self.top_surface_tension * 2.0 * (std::f64::consts::PI * pos[1] as f64).sqrt()
                - self.top_driving_force * pos[1] as f64;
        } else if pos[0] == 1 {
            return self.bot_surface_tension * 2.0 * (std::f64::consts::PI * pos[1] as f64).sqrt()
                - self.bot_driving_force * pos[1] as f64;
        }
        panic!("pos[0] should be 0 or 1, but was {}", pos[0]);
    }
}

impl MarkovChain<u32, BasicReaction<u32>> for CNTLadderChain {
    fn initialize(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition<u32>>,
    ) {
    }

    fn on_reaction(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition<u32>>,
        _reaction_id: usize,
        _dt: f64,
    ) {
    }

    fn num_possible_reactions(&self, state: &Array2<u32>) -> usize {
        6 * state.shape()[1] - 4
    }

    fn reaction(
        &self,
        state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition<u32>>,
        mut reaction_id: usize,
    ) -> BasicReaction<u32> {
        let length = state.shape()[1];
        if reaction_id < 2 * length - 2 {
            let from = [reaction_id % 2, reaction_id / 2];
            let to = [reaction_id % 2, reaction_id / 2 + 1];
            return BasicReaction::Diffusion { from, to };
        }
        reaction_id -= 2 * length - 2;
        if reaction_id < 2 * length - 2 {
            let from = [reaction_id % 2, reaction_id / 2 + 1];
            let to = [reaction_id % 2, reaction_id / 2];
            return BasicReaction::Diffusion { from, to };
        }
        reaction_id -= 2 * length - 2;
        if reaction_id < 2 * length {
            let from = [reaction_id % 2, reaction_id / 2];
            let to = [(reaction_id + 1) % 2, reaction_id / 2];
            return BasicReaction::Diffusion { from, to };
        }
        panic!("Reaction id {} is too high!", reaction_id + 4 * length - 4)
    }

    fn rate(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition<u32>>,
        reaction_id: usize,
    ) -> f64 {
        use BasicReaction as BR;
        let reaction = self.reaction(state, boundary, reaction_id);
        if !self.is_possible(state, reaction) {
            return 0.0;
        }
        match reaction {
            BR::Diffusion {
                from: from @ [1, _],
                to: to @ [1, _],
            }
            | BR::Diffusion {
                from: from @ [0, _],
                to: to @ [0, _],
            } => (self.beta / 2.0 * (self.free_energy(from) - self.free_energy(to))).exp(),
            BR::Diffusion {
                from: [1, _],
                to: [0, _],
            } => self.rescue_rate,
            BR::Diffusion {
                from: [0, _],
                to: [1, _],
            } => self.off_rate,

            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }

    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition<u32>>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]> {
        let reaction = self.reaction(state, boundary, reaction_id);
        match reaction {
            BasicReaction::Diffusion { from, to: _to } => vec![from],
            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }
}
