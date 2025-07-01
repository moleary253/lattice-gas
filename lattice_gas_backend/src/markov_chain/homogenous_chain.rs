use super::*;
use crate::reaction::BasicReaction;
use serde::{Deserialize, Serialize};

pub const EMPTY: u32 = 0;
pub const INERT: u32 = 1;
pub const BONDING: u32 = 2;

/// Describes a homogenous bonding to inert reaction as described in Yeongik's PRL in 2023.
/// Sets $D$ to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct HomogenousChain {
    #[pyo3(get)]
    pub beta: f64,
    #[pyo3(get)]
    pub bond_energy: f64,
    #[pyo3(get)]
    pub driving_chemical_potential: f64,

    #[pyo3(get)]
    pub inert_fugacity: f64,
    #[pyo3(get)]
    pub bonding_fugacity: f64,
    #[pyo3(get)]
    pub inert_to_bonding_rate: f64,
}

#[pymethods]
impl HomogenousChain {
    #[new]
    pub fn new(
        beta: f64,
        bond_energy: f64,
        driving_chemical_potential: f64,
        inert_fugacity: f64,
        bonding_fugacity: f64,
        inert_to_bonding_rate: f64,
    ) -> Self {
        HomogenousChain {
            beta,
            bond_energy,
            driving_chemical_potential,
            inert_fugacity,
            bonding_fugacity,
            inert_to_bonding_rate,
        }
    }
}

impl HomogenousChain {
    pub fn site_energy(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        position: [usize; 2],
    ) -> f64 {
        let mut energy = 0.;
        for adjacent_state in boundary.adjacent(&state, position) {
            if adjacent_state == 2 {
                energy += self.bond_energy;
            }
        }
        energy
    }

    pub fn delta_f_res(&self) -> f64 {
        (self.inert_fugacity / self.bonding_fugacity).ln() / self.beta
    }

    pub fn is_possible(&self, state: &ArrayView2<u32>, reaction: BasicReaction<u32>) -> bool {
        match reaction {
            BasicReaction::PointChange {
                from,
                to: _to,
                position,
            } => state[position] == from,
            _ => false,
        }
    }
}

#[typetag::serde]
impl MarkovChain for HomogenousChain {
    fn initialize(&mut self, _state: &ArrayView2<u32>, _boundary: &Box<dyn BoundaryCondition>) {}

    fn on_reaction(
        &mut self,
        _state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        _reaction_id: usize,
        _dt: f64,
    ) {
    }

    fn num_possible_reactions(&self, state: &Array2<u32>) -> usize {
        state.len() * 6
    }

    fn reaction(
        &self,
        state: &ArrayView2<u32>,
        _boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> BasicReaction<u32> {
        let possible_states = [EMPTY, INERT, BONDING];
        let position_index = reaction_id / 6;
        let position = [
            position_index / state.shape()[1],
            position_index % state.shape()[1],
        ];
        let from = possible_states[reaction_id % 3];
        let mut i = 0;
        let mut j = 0;
        let to = loop {
            if possible_states[i] == from {
                i += 1;
                continue;
            }
            if j == reaction_id % 2 {
                break possible_states[i];
            }
            i += 1;
            j += 1;
        };
        BasicReaction::PointChange { from, to, position }
    }

    fn rate(
        &self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> f64 {
        use BasicReaction as BR;
        let reaction = self.reaction(state, boundary, reaction_id);
        if !self.is_possible(state, reaction) {
            return 0.0;
        }
        match reaction {
            BR::PointChange {
                from: EMPTY,
                to: INERT,
                position: _,
            } => self.inert_fugacity,
            BR::PointChange {
                from: EMPTY,
                to: BONDING,
                position: _,
            } => self.bonding_fugacity,

            BR::PointChange {
                from: INERT,
                to: EMPTY,
                position: _,
            } => 1.,
            BR::PointChange {
                from: INERT,
                to: BONDING,
                position: _,
            } => self.inert_to_bonding_rate,

            BR::PointChange {
                from: BONDING,
                to: EMPTY,
                position,
            } => (self.beta * self.site_energy(state, boundary, position)).exp(),
            BR::PointChange {
                from: BONDING,
                to: INERT,
                position,
            } => {
                self.inert_to_bonding_rate * self.inert_fugacity / self.bonding_fugacity
                    * (self.beta
                        * (self.site_energy(state, boundary, position)
                            + self.driving_chemical_potential))
                        .exp()
            }

            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }

    fn indicies_affecting_reaction(
        &mut self,
        state: &ArrayView2<u32>,
        boundary: &Box<dyn BoundaryCondition>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]> {
        let reaction = self.reaction(state, boundary, reaction_id);
        match reaction {
            BasicReaction::PointChange { from, to, position } => match (from, to) {
                (BONDING, _) | (_, BONDING) => {
                    let mut reactions = boundary.adjacent_indicies(&state, position);
                    reactions.push(position);
                    reactions
                }
                _ => vec![position],
            },
            _ => panic!("Unexpected reaction {:?}", reaction),
        }
    }
}
