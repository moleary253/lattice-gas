use super::*;
use crate::reaction::BasicReaction;
use crate::SiteState;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Describes a homogenous bonding to inert reaction as described in Yeongik's PRL in 2023.
/// Sets $D$ to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomogenousChain {
    pub beta: f64,
    pub bond_energy: f64,
    pub driving_chemical_potential: f64,

    pub inert_fugacity: f64,
    pub bonding_fugacity: f64,
    pub inert_to_bonding_rate: f64,
}

impl HomogenousChain {
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
        state: &Array2<SiteState>,
        boundary: &impl BoundaryCondition<SiteState>,
        position: [usize; 2],
    ) -> f64 {
        let mut energy = 0.;
        for adjacent_state in boundary.adjacent(&state, position) {
            if adjacent_state == SiteState::Bonding {
                energy += self.bond_energy;
            }
        }
        energy
    }

    pub fn delta_f_res(&self) -> f64 {
        (self.inert_fugacity / self.bonding_fugacity).ln() / self.beta
    }

    pub fn is_possible(
        &self,
        state: &Array2<SiteState>,
        reaction: BasicReaction<SiteState>,
    ) -> bool {
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

impl MarkovChain<SiteState, BasicReaction<SiteState>> for HomogenousChain {
    fn initialize(
        &mut self,
        _state: &Array2<SiteState>,
        _boundary: &impl BoundaryCondition<SiteState>,
    ) {
    }

    fn on_reaction(
        &mut self,
        _state: &Array2<SiteState>,
        _boundary: &impl BoundaryCondition<SiteState>,
        _reaction_id: usize,
        _dt: f64,
    ) {
    }

    fn num_possible_reactions(&self, state: &Array2<SiteState>) -> usize {
        state.len() * 6
    }

    fn reaction(&self, state: &Array2<SiteState>, reaction_id: usize) -> BasicReaction<SiteState> {
        let possible_states = [SiteState::Empty, SiteState::Inert, SiteState::Bonding];
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
        state: &Array2<SiteState>,
        boundary: &impl BoundaryCondition<SiteState>,
        reaction_id: usize,
    ) -> f64 {
        use BasicReaction as BR;
        use SiteState as SS;
        let reaction = self.reaction(state, reaction_id);
        if !self.is_possible(state, reaction) {
            return 0.0;
        }
        match reaction {
            BR::PointChange {
                from: SS::Empty,
                to: SS::Inert,
                position: _,
            } => self.inert_fugacity,
            BR::PointChange {
                from: SS::Empty,
                to: SS::Bonding,
                position: _,
            } => self.bonding_fugacity,

            BR::PointChange {
                from: SS::Inert,
                to: SS::Empty,
                position: _,
            } => 1.,
            BR::PointChange {
                from: SS::Inert,
                to: SS::Bonding,
                position: _,
            } => self.inert_to_bonding_rate,

            BR::PointChange {
                from: SS::Bonding,
                to: SS::Empty,
                position,
            } => (self.beta * self.site_energy(state, boundary, position)).exp(),
            BR::PointChange {
                from: SS::Bonding,
                to: SS::Inert,
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
        state: &Array2<SiteState>,
        boundary: &impl BoundaryCondition<SiteState>,
        reaction_id: usize,
    ) -> Vec<[usize; 2]> {
        let reaction = self.reaction(state, reaction_id);
        match reaction {
            BasicReaction::PointChange { from, to, position } => match (from, to) {
                (SiteState::Bonding, _) | (_, SiteState::Bonding) => {
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
