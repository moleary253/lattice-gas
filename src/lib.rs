use ndarray::Array2;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SiteState {
    Bonding,
    Inert,
    Empty,
}

impl Default for SiteState {
    fn default() -> Self {
        SiteState::Empty
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Reaction<T>
where
    T: Clone,
{
    Diffusion {
        from: [usize; 2],
        to: [usize; 2],
    },
    PointChange {
        from: T,
        to: T,
        position: [usize; 2],
    },
}

impl<T> Reaction<T>
where
    T: Clone,
{
    pub fn point_change(from: T, to: T, position: [usize; 2]) -> Self {
        Reaction::PointChange { from, to, position }
    }

    pub fn diffusion(to: [usize; 2], from: [usize; 2]) -> Self {
        Reaction::Diffusion { from, to }
    }
}

pub trait MarkovChain {
    fn rate(&self, system: &System, reaction: Reaction<SiteState>) -> f64;
    fn allowed_reactions(&self, system: &System) -> Vec<Reaction<SiteState>>;
}

/*
 * Describes an Ising model.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsingChain {
    pub beta: f64,
    pub bond_energy: f64, // J
}

impl IsingChain {
    pub fn site_energy(&self, system: &System, position: [usize; 2]) -> f64 {
        let mut energy = 0.;
        let [x, y] = position;
        let (x, y) = (x as i32, y as i32);
        for (dx, dy) in [(-1, 0), (1, 0), (0, 1), (0, -1)] {
            let adjacent_state = system.state[[
                (dx + x).rem_euclid(system.state.shape()[0] as i32) as usize,
                (dy + y).rem_euclid(system.state.shape()[0] as i32) as usize,
            ]];
            if adjacent_state == SiteState::Bonding {
                energy += self.bond_energy;
            } else if adjacent_state == SiteState::Empty {
                energy -= self.bond_energy;
            }
        }
        energy
    }
}

impl MarkovChain for IsingChain {
    fn rate(&self, system: &System, reaction: Reaction<SiteState>) -> f64 {
        match reaction {
            Reaction::PointChange { from, to, position } => match (from, to) {
                (SiteState::Empty, SiteState::Bonding) => (8. * self.beta * self.bond_energy).exp(),
                (SiteState::Bonding, SiteState::Empty) => (8. * self.beta * self.bond_energy
                    + 2. * self.beta * self.site_energy(system, position))
                .exp(),
                _ => 0.,
            },
            _ => 0.,
        }
    }

    fn allowed_reactions(&self, system: &System) -> Vec<Reaction<SiteState>> {
        let mut reactions = Vec::with_capacity(system.state.len() * 2);
        for (i, state) in system.state.iter().enumerate() {
            let pos = system.pos_of_ith_site(i);

            for next_state in [SiteState::Empty, SiteState::Inert] {
                if *state == next_state {
                    continue;
                }
                reactions.push(Reaction::point_change(*state, next_state, pos));
            }
        }

        return reactions;
    }
}

/*
 * Describes a homogenous bonding to inert reaction as described in Yeongik's PRL in 2023.
 * Sets $D$ to 1.
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomogenousChain {
    pub beta: f64,
    pub bond_energy: f64,
    pub driving_chemical_potential: f64,

    pub inert_fugacity: f64,
    pub bonding_fugacity: f64,
    pub inert_to_bonding_rate: f64,
}

/*
 * Constructors
 */
impl HomogenousChain {
    pub fn beta_1(
        bond_energy: f64,
        driving_chemical_potential: f64,
        inert_fugacity: f64,
        bonding_fugacity: f64,
        inert_to_bonding_rate: f64,
    ) -> Self {
        HomogenousChain {
            beta: 1.,
            bond_energy,
            driving_chemical_potential,
            inert_fugacity,
            bonding_fugacity,
            inert_to_bonding_rate,
        }
    }

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
    pub fn site_energy(&self, system: &System, position: [usize; 2]) -> f64 {
        let mut energy = 0.;
        let [x, y] = position;
        let (x, y) = (x as i32, y as i32);
        for (dx, dy) in [(-1, 0), (1, 0), (0, 1), (0, -1)] {
            let adjacent_state = system.state[[
                (dx + x).rem_euclid(system.state.shape()[0] as i32) as usize,
                (dy + y).rem_euclid(system.state.shape()[1] as i32) as usize,
            ]];
            if adjacent_state == SiteState::Bonding {
                energy += self.bond_energy;
            }
        }
        energy
    }

    pub fn delta_f_res(&self) -> f64 {
        (self.inert_fugacity / self.bonding_fugacity).ln() / self.beta
    }
}

impl MarkovChain for HomogenousChain {
    fn rate(&self, system: &System, reaction: Reaction<SiteState>) -> f64 {
        match reaction {
            Reaction::PointChange { from, to, position } => match (from, to) {
                (SiteState::Empty, SiteState::Inert) => self.inert_fugacity,
                (SiteState::Empty, SiteState::Bonding) => self.bonding_fugacity,

                (SiteState::Inert, SiteState::Empty) => 1.,
                (SiteState::Inert, SiteState::Bonding) => self.inert_to_bonding_rate,

                (SiteState::Bonding, SiteState::Empty) => {
                    (self.beta * self.site_energy(system, position)).exp()
                }
                (SiteState::Bonding, SiteState::Inert) => {
                    self.inert_to_bonding_rate * self.inert_fugacity / self.bonding_fugacity
                        * (self.beta
                            * (self.site_energy(system, position)
                                + self.driving_chemical_potential))
                            .exp()
                }

                _ => 0.,
            },
            _ => 0.,
        }
    }

    fn allowed_reactions(&self, system: &System) -> Vec<Reaction<SiteState>> {
        let mut reactions = Vec::with_capacity(system.state.len() * 2);
        for (i, state) in system.state.iter().enumerate() {
            let pos = system.pos_of_ith_site(i);

            for next_state in [SiteState::Empty, SiteState::Inert, SiteState::Bonding] {
                if *state == next_state {
                    continue;
                }
                reactions.push(Reaction::point_change(*state, next_state, pos));
            }
        }

        return reactions;
    }
}

pub struct System {
    pub state: Array2<SiteState>,
    pub time: f64,

    pub chain: Box<dyn MarkovChain>,
}

/*
 * Constructors
 */
impl System {
    pub fn empty(width: usize, height: usize, chain: Box<dyn MarkovChain>) -> Self {
        System {
            state: Array2::default((height, width)),
            time: 0.,
            chain,
        }
    }

    pub fn full(
        width: usize,
        height: usize,
        chain: Box<dyn MarkovChain>,
        state: SiteState,
    ) -> Self {
        System {
            state: Array2::from_elem((height, width), state),
            time: 0.,
            chain,
        }
    }

    pub fn with_state(chain: Box<dyn MarkovChain>, state: Array2<SiteState>) -> Self {
        System {
            state,
            time: 0.,
            chain,
        }
    }
}

/*
 * Access to data
 */
impl System {
    pub fn pos_of_ith_site(&self, i: usize) -> [usize; 2] {
        [i / self.state.shape()[1], i % self.state.shape()[1]]
    }

    pub fn ith_site(&self, i: usize) -> SiteState {
        self.state[self.pos_of_ith_site(i)]
    }

    pub fn set_state(&mut self, time: f64, state: Array2<SiteState>) {
        self.time = time;
        self.state = state;
    }
}

/*
 * Evolution
 */
impl System {
    pub fn next_reaction(&self) -> (f64, Reaction<SiteState>) {
        let mut partial_sums_of_rates: Vec<f64> = Vec::with_capacity(self.state.len());
        for (i, state) in self.state.iter().enumerate() {
            let pos = self.pos_of_ith_site(i);

            partial_sums_of_rates.push(*partial_sums_of_rates.last().unwrap_or(&0.));
            partial_sums_of_rates[i] += self
                .chain
                .rate(&self, Reaction::new(*state, SiteState::Empty, pos));
            partial_sums_of_rates[i] += self
                .chain
                .rate(&self, Reaction::new(*state, SiteState::Inert, pos));
            partial_sums_of_rates[i] += self
                .chain
                .rate(&self, Reaction::new(*state, SiteState::Bonding, pos));
        }

        let mut rng = rand::thread_rng();
        let tau = -(rng.gen::<f64>()).ln() / partial_sums_of_rates.last().unwrap();
        let chosen_partial_sum = rng.gen::<f64>() * partial_sums_of_rates.last().unwrap();

        let chosen_i = (&partial_sums_of_rates).partition_point(|a| *a <= chosen_partial_sum);
        let chosen_pos = self.pos_of_ith_site(chosen_i);
        let chosen_state = self.state[chosen_pos];

        let next_state = {
            let mut sum = *partial_sums_of_rates.get(chosen_i - 1).unwrap_or(&0.);
            let states = [SiteState::Empty, SiteState::Inert, SiteState::Bonding];
            let mut states_iter = states.iter();
            loop {
                let state = states_iter.next().unwrap();
                sum += self.chain.rate(
                    &self,
                    Reaction::point_change(chosen_state, *state, chosen_pos),
                );
                if sum > chosen_partial_sum {
                    break *state;
                }
            }
        };

        (
            tau,
            Reaction::point_change(self.state[chosen_pos], next_state, chosen_pos),
        )
    }

    pub fn update(&mut self, delta_t: f64, reaction: Reaction<SiteState>) {
        self.time += delta_t;
        match reaction {
            Reaction::PointChange {
                from: _,
                to,
                position,
            } => self.state[position] = to,
            Reaction::Diffusion { from, to } => {
                (self.state[from], self.state[to]) = (self.state[to], self.state[from]);
            }
        }
    }

    pub fn simulate_one_step_inplace(&mut self) {
        let (delta_t, reaction) = self.next_reaction();
        self.update(delta_t, reaction);
    }
}

/*
 * System-wide statistics
 */
impl System {
    pub fn particle_number(&self) -> HashMap<SiteState, usize> {
        let mut counts = HashMap::new();
        for site in self.state.iter() {
            counts.entry(*site).and_modify(|n| *n += 1).or_insert(1);
        }
        counts
    }
}

pub fn update_running_average(
    system: &System,
    delta_t: f64,
    averaging_interval: f64,
    running_sum: &mut f64,
    time_measured: &mut f64,
    measurement: &dyn Fn(&System) -> f64,
) -> Option<f64> {
    if *time_measured + delta_t < averaging_interval {
        *time_measured += delta_t;
        *running_sum += delta_t * measurement(system);
        return None;
    }

    let measured = measurement(system);
    *running_sum += (averaging_interval - *time_measured) * measured;

    let average = *running_sum / averaging_interval;

    *running_sum = (*time_measured + delta_t - averaging_interval) * measured;
    *time_measured = *time_measured + delta_t - averaging_interval;

    Some(average)
}
