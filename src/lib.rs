use ndarray::Array2;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod analysis;
pub mod binary_sum_tree;
pub mod boundary_condition;
pub mod ending_criterion;
pub mod markov_chain;
pub mod reaction;
pub mod serialize;
pub mod simulate;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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

pub trait MarkovChain<T>
where
    T: Clone,
{
    fn rate(&self, system: &System, reaction: Reaction<T>) -> f64;
    fn allowed_reactions(&self, system: &System) -> Vec<Reaction<T>>;
}

/// A statistic which can be calculated on a sytem.
///
/// The goal is that this caches the value it calculated to make reacalculating
/// the statistic easier.
pub trait Statistic<T> {
    /// Initialize the statistic on a system
    fn initialize(&mut self, system: &System);

    /// Update the cached statistic when a new reaction happens
    fn update(&mut self, system_pre_reaction: &System, delta_t: f64, reaction: Reaction<SiteState>);

    /// Retrieve the value of the statistic
    fn value(&self) -> &T;
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

impl MarkovChain<SiteState> for IsingChain {
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

/// Describes a Markov chain which is meant to model T-Cells.
///
/// Assumptions:
/// - Zeta-chain / Zap70 system can be modeled well by a Gaussian approximation to a WLC
///     - Neglects excluded volume with cell wall anywhere on the WLC except at the end.
/// - Lck recruitment is not important
/// - Zap70 concentration is not time dependent
/// - Zap70 + LAT reaction involves interactions between one LAT and one Zap70
/// - Reaction size is very small for Zap70-LAT reaction
/// - Right now, diffusion limited, NOT bond limited
///
/// Allowed Reactions:
/// - Diffusion with rate constant 1
/// - pLAT(B) to LAT(I) & reverse, catalyzed by Zap70, pushing towards pLAT with Delta mu
///     - With Michaelis-Mentin kinetics
/// - pLAT(B) to LAT(I) & reverse, catalyzed by phosphatase, undriven
///     - With Michaelis-Mentin kinetics
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCellChain {
    pub beta: f64,

    pub bond_energy: f64,

    pub kinase_chemical_potential: f64,
    pub kinase_rate_constant: f64,
    pub kinase_michaelis_constant: f64,

    pub zeta_chain_extension_stdev: f64,
    pub t_cell_receptor_position: [usize; 2],

    pub phosphatase_chemical_potential: f64,
    pub phosphatase_rate_constant: f64,
    pub phosphatase_michaelis_constant: f64,

    kinase_concentration: Option<Array2<f64>>,
    energy: Option<Array2<i32>>,
    concentration_near_receptor: Option<f64>,
}

/*
 * Constructors
 */
impl TCellChain {
    pub fn new(
        beta: f64,
        bond_energy: f64,
        kinase_chemical_potential: f64,
        kinase_rate_constant: f64,
        kinase_michaelis_constant: f64,
        zeta_chain_extension_stdev: f64,
        t_cell_receptor_position: [usize; 2],
        phosphatase_chemical_potential: f64,
        phosphatase_rate_constant: f64,
        phosphatase_michaelis_constant: f64,
    ) -> Self {
        TCellChain {
            beta,
            bond_energy,
            kinase_chemical_potential,
            kinase_rate_constant,
            kinase_michaelis_constant,
            zeta_chain_extension_stdev,
            t_cell_receptor_position,
            phosphatase_chemical_potential,
            phosphatase_rate_constant,
            phosphatase_michaelis_constant,
            kinase_concentration: None,
            energy: None,
            concentration_near_receptor: None,
        }
    }
}

impl TCellChain {
    pub fn site_energy(&self, system: &System, position: [usize; 2]) -> f64 {
        match &self.energy {
            Some(energy) => {
                return energy[position] as f64 * self.bond_energy;
            }
            None => {}
        }
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

    pub fn kinase_probability(&self, position: [usize; 2]) -> f64 {
        match &self.kinase_concentration {
            Some(kinase_concentration) => {
                return kinase_concentration[position];
            }
            None => {}
        }
        let [x, y] = position;
        let (x, y) = (x as f64, y as f64);
        let (dx, dy) = (
            x - self.t_cell_receptor_position[0] as f64,
            y - self.t_cell_receptor_position[1] as f64,
        );
        let distance_squared = dx * dx + dy * dy;

        (3. / 2. / std::f64::consts::PI / self.zeta_chain_extension_stdev.powi(2))
            .powi(3)
            .sqrt()
            * (3. * distance_squared / 2. / self.zeta_chain_extension_stdev.powi(2)).exp()
    }

    pub fn substrate_concentration_near_receptor(&self, system: &System) -> f64 {
        match self.concentration_near_receptor {
            Some(concentration) => {
                return concentration;
            }
            None => {}
        }

        let mut concentration = 0.0;
        for (i, state) in system.state.iter().enumerate() {
            if *state == SiteState::Empty {
                continue;
            }

            let pos = system.pos_of_ith_site(i);

            concentration += self.kinase_probability(pos)
        }

        return concentration;
    }

    pub fn initialize_cache(&mut self, system: &System) {
        let mut energy = Array2::zeros(system.state.raw_dim());
        let mut kinase_concentration = Array2::zeros(system.state.raw_dim());

        for (i, state) in system.state.iter().enumerate() {
            let pos = system.pos_of_ith_site(i);
            kinase_concentration[pos] = self.kinase_probability(pos);

            if *state != SiteState::Bonding {
                continue;
            }

            for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                let adjacent_pos = [
                    (pos[0] as i32 + dx) as usize % system.state.shape()[0],
                    (pos[1] as i32 + dy) as usize % system.state.shape()[1],
                ];
                energy[adjacent_pos] += 1;
            }
        }

        self.energy = Some(energy);
        self.kinase_concentration = Some(kinase_concentration);
        self.concentration_near_receptor = Some(self.substrate_concentration_near_receptor(system));
    }

    pub fn update_cache(&mut self, system: &System, _delta_t: f64, reaction: Reaction<SiteState>) {
        let energy: &mut Array2<i32> = match &mut self.energy {
            Some(energy) => energy,
            None => {
                return;
            }
        };

        match reaction {
            Reaction::PointChange {
                from,
                to,
                position: pos,
            } => {
                let change = if from == SiteState::Bonding {
                    1
                } else if to == SiteState::Bonding {
                    -1
                } else {
                    return;
                };
                for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                    let adjacent_pos = [
                        (pos[0] as i32 + dx) as usize % system.state.shape()[0],
                        (pos[1] as i32 + dy) as usize % system.state.shape()[1],
                    ];
                    energy[adjacent_pos] += change;
                }
            }
            Reaction::Diffusion { from, to } => {
                let change = if system.state[from] == SiteState::Bonding {
                    1
                } else if system.state[to] == SiteState::Bonding {
                    -1
                } else {
                    return;
                };
                for (pos, multiplier) in [(from, -1), (to, 1)] {
                    for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
                        let adjacent_pos = [
                            (pos[0] as i32 + dx) as usize % system.state.shape()[0],
                            (pos[1] as i32 + dy) as usize % system.state.shape()[1],
                        ];
                        energy[adjacent_pos] += change * multiplier;
                    }
                }
            }
        }
    }
}

impl MarkovChain<SiteState> for TCellChain {
    fn rate(&self, system: &System, reaction: Reaction<SiteState>) -> f64 {
        match reaction {
            Reaction::PointChange { from, to, position } => match (from, to) {
                (SiteState::Inert, SiteState::Bonding) => {
                    self.kinase_rate_constant * self.kinase_probability(position)
                        / (self.kinase_michaelis_constant
                            + self.substrate_concentration_near_receptor(system))
                        + self.phosphatase_rate_constant
                            * (self.beta
                                * (self.site_energy(system, position)
                                    + self.phosphatase_chemical_potential))
                                .exp()
                }

                (SiteState::Bonding, SiteState::Inert) => {
                    self.kinase_rate_constant * self.kinase_probability(position)
                        / (self.kinase_michaelis_constant
                            + self.substrate_concentration_near_receptor(system))
                        * (self.beta
                            * (self.site_energy(system, position) + self.kinase_chemical_potential))
                            .exp()
                        + self.phosphatase_rate_constant
                }
                _ => 0.,
            },
            Reaction::Diffusion { from, to } => {
                if (from[0] as i64 - to[0] as i64).abs() + (from[1] as i64 - to[1] as i64).abs()
                    != 1
                {
                    return 0.;
                }

                if system.state[from] == system.state[to] {
                    return 0.;
                }

                if system.state[to] == SiteState::Bonding {
                    return (self.beta
                        * (self.site_energy(system, to) - self.site_energy(system, from)))
                    .exp();
                }

                if system.state[from] == SiteState::Bonding {
                    return (self.beta
                        * (self.site_energy(system, from) - self.site_energy(system, to)))
                    .exp();
                }

                1.
            }
        }
    }

    fn allowed_reactions(&self, system: &System) -> Vec<Reaction<SiteState>> {
        let mut reactions = Vec::with_capacity(system.state.len() * 3);
        for (i, state) in system.state.iter().enumerate() {
            let pos = system.pos_of_ith_site(i);
            for (dx, dy) in [(0, 1), (1, 0)] {
                let adjacent_pos = [
                    (pos[0] + dx) % system.state.shape()[0],
                    (pos[1] + dy) % system.state.shape()[1],
                ];
                if system.state[pos] == system.state[adjacent_pos] {
                    continue;
                }
                reactions.push(Reaction::diffusion(
                    pos,
                    [
                        (pos[0] + dx) % system.state.shape()[0],
                        (pos[1] + dy) % system.state.shape()[1],
                    ],
                ));
            }

            if *state == SiteState::Empty {
                continue;
            }

            for next_state in [SiteState::Inert, SiteState::Bonding] {
                if *state == next_state {
                    continue;
                }
                reactions.push(Reaction::point_change(*state, next_state, pos));
            }
        }

        return reactions;
    }
}

/// Keeps track of the number of each type of particle in the system.
pub struct ParticleNumberStatistic {
    counts: HashMap<SiteState, usize>,
}

impl ParticleNumberStatistic {
    pub fn new() -> ParticleNumberStatistic {
        ParticleNumberStatistic {
            counts: HashMap::new(),
        }
    }
}

impl Statistic<HashMap<SiteState, usize>> for ParticleNumberStatistic {
    fn initialize(&mut self, system: &System) {
        self.counts = HashMap::new();
        for site in system.state.iter() {
            self.counts
                .entry(*site)
                .and_modify(|n| *n += 1)
                .or_insert(1);
        }
    }

    fn update(&mut self, _system: &System, _delta_t: f64, reaction: Reaction<SiteState>) {
        match reaction {
            Reaction::PointChange {
                from,
                to,
                position: _,
            } => {
                *self.counts.get_mut(&from).unwrap() -= 1;
                self.counts
                    .insert(to, *self.counts.get(&to).unwrap_or(&0) + 1);
            }
            _ => {}
        }
    }

    fn value(&self) -> &HashMap<SiteState, usize> {
        &self.counts
    }
}

pub struct System {
    pub state: Array2<SiteState>,
    pub time: f64,
}

/*
 * Constructors
 */
impl System {
    pub fn empty(width: usize, height: usize) -> Self {
        System {
            state: Array2::default((height, width)),
            time: 0.,
        }
    }

    pub fn full(width: usize, height: usize, state: SiteState) -> Self {
        System {
            state: Array2::from_elem((height, width), state),
            time: 0.,
        }
    }

    pub fn random(width: usize, height: usize, odds: [(SiteState, f64); 3]) -> Self {
        let mut state = Array2::default((height, width));

        let mut rng = rand::rng();
        let sum_of_weights = odds.iter().fold(0., |sum, (_, w)| sum + w);
        for site in state.iter_mut() {
            let rand_number = rng.random::<f64>() * sum_of_weights;
            let mut sum = 0.;
            for (state, weight) in odds {
                sum += weight;
                if sum >= rand_number {
                    *site = state;
                    break;
                }
            }
        }
        System { state, time: 0. }
    }

    pub fn with_state(state: Array2<SiteState>) -> Self {
        System { state, time: 0. }
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
    pub fn next_reaction(
        &self,
        chain: &dyn MarkovChain<SiteState>,
        rng: &mut impl Rng,
    ) -> (f64, Reaction<SiteState>) {
        let reactions = chain.allowed_reactions(&self);
        let mut partial_sums_of_rates: Vec<f64> = Vec::with_capacity(reactions.len());
        let mut sum = 0.;
        for reaction in reactions.iter() {
            sum += chain.rate(&self, *reaction);
            partial_sums_of_rates.push(sum);
        }
        let partial_sums_of_rates = partial_sums_of_rates;

        let tau = -(rng.random::<f64>()).ln() / partial_sums_of_rates.last().unwrap();
        let chosen_partial_sum = rng.random::<f64>() * partial_sums_of_rates.last().unwrap();

        let chosen_reaction =
            reactions[(&partial_sums_of_rates).partition_point(|a| *a <= chosen_partial_sum)];

        (tau, chosen_reaction)
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

    pub fn simulate_one_step_inplace(
        &mut self,
        chain: &dyn MarkovChain<SiteState>,
        rng: &mut impl Rng,
    ) {
        let (delta_t, reaction) = self.next_reaction(chain, rng);
        self.update(delta_t, reaction);
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
