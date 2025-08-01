use crate::analysis::Analyzer;
use crate::binary_sum_tree::BinarySumTree;
use crate::boundary_condition::BoundaryCondition;
use crate::ending_criterion::EndingCriterion;
use crate::markov_chain::MarkovChain;
use crate::reaction::{BasicReaction, Reaction};
use numpy::ndarray::Array2;
use rand::prelude::*;

use crate::serialize;

use numpy::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;

// (MLO) Adding functionality to save npy files

use ndarray_npy::write_npy;

// Struct to hold simulation state
struct SimulationState {
    pub state: Array2<u32>,
    pub boundary: Box<dyn BoundaryCondition>,
    pub chain: Box<dyn MarkovChain>,
    pub analyzers: Vec<Box<dyn Analyzer>>,
    pub ending_criteria: Vec<Box<dyn EndingCriterion>>,
    pub rates: BinarySumTree<f64>,
    pub reactions_depending_on_locations: Array2<Vec<usize>>,
    pub reactions: Vec<BasicReaction<u32>>,
    pub delta_times: Vec<f64>,
}

#[pyfunction]
#[pyo3(name = "simulate")]
/// Runs a KMC simulation.
///
/// Arguments:
///   - initial_state: The initial conditions of the simulation. Should be 2D
///     numpy array with dtype `"u4"`.
///   - boundary: The boundary conditions of the simulation. Should be a rust
///     object which implements BoundaryCondition<u32>.
///   - chain: The Markov chain that governs transition rates. Should be a rust
///     object which implements MarkovChain.
///   - analyzers: A list of `Analyzer` objects to run analysis during the
///     simulation.
///   - ending_criteria: Determine when the simulation should end. Should be a
///     python list of rust objects which implement EndingCriteron.
///   - seed: The random seed to initialize the rng.
///   - output_file: Where to save the simulation results. Should be a string
///     which points to the location of a .tar.gz file.
pub fn py_simulate(
    py: Python<'_>,
    initial_state: &Bound<'_, PyArray2<u32>>,
    boundary: &Bound<'_, PyAny>,
    chain: &Bound<'_, PyAny>,
    analyzers: Vec<Box<dyn Analyzer>>,
    ending_criteria: Vec<Bound<'_, PyAny>>,
    seed: u64,
    output_dir: &str,
) -> PyResult<()> {
    let mut sim_state =
        py_initialize_simulation(initial_state, boundary, chain, analyzers, ending_criteria)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let initial_conditions = sim_state.state.clone();

    loop {
        let should_end = simulate_iteration(&mut sim_state, &mut rng);
        if should_end {
            break;
        }
        py.check_signals()?;
    }

    if let Err(err) = serialize::save_simulation(
        output_dir,
        &sim_state.chain,
        &sim_state.boundary,
        &sim_state.ending_criteria,
        &initial_conditions,
        &sim_state.analyzers,
        &sim_state.delta_times,
        &sim_state.state,
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            err.to_string(),
        ));
    }

    // (MLO) Adding functionality to save npy files
    // There is probably a better way to do this, within the save_simulation function, I have to look into it.

    let final_state_path = format!("{}/final_state.npy", output_dir);
    write_npy(&final_state_path, &sim_state.state).map_err(|err| PyErr::new::<pyo3::exceptions::PyIOError, _>(err.to_string()))?;

    Ok(())
}

pub fn simulate(
    state: Array2<u32>,
    boundary: Box<dyn BoundaryCondition>,
    chain: Box<dyn MarkovChain>,
    analyzers: Vec<Box<dyn Analyzer>>,
    ending_criteria: Vec<Box<dyn EndingCriterion>>,
    mut rng: impl Rng,
) -> (
    Array2<u32>,
    Vec<f64>,
    Vec<BasicReaction<u32>>,
    Vec<Box<dyn Analyzer>>,
) {
    let mut sim_state = initialize_simulation(state, boundary, chain, analyzers, ending_criteria);

    loop {
        let should_end = simulate_iteration(&mut sim_state, &mut rng);
        if should_end {
            break;
        }
    }

    (
        sim_state.state,
        sim_state.delta_times,
        sim_state.reactions,
        sim_state.analyzers,
    )
}

fn py_initialize_simulation(
    state: &Bound<'_, PyArray2<u32>>,
    boundary: &Bound<'_, PyAny>,
    chain: &Bound<'_, PyAny>,
    analyzers: Vec<Box<dyn Analyzer>>,
    ending_criteria: Vec<Bound<'_, PyAny>>,
) -> PyResult<SimulationState> {
    let state = state.to_owned_array();
    let boundary = boundary.extract()?;
    let chain = chain.extract()?;

    let mut rust_ending_criteria = Vec::with_capacity(ending_criteria.len());
    for ending_criterion in ending_criteria.iter() {
        rust_ending_criteria.push(ending_criterion.extract()?);
    }
    Ok(initialize_simulation(
        state,
        boundary,
        chain,
        analyzers,
        rust_ending_criteria,
    ))
}

fn initialize_simulation(
    state: Array2<u32>,
    boundary: Box<dyn BoundaryCondition>,
    mut chain: Box<dyn MarkovChain>,
    analyzers: Vec<Box<dyn Analyzer>>,
    mut ending_criteria: Vec<Box<dyn EndingCriterion>>,
) -> SimulationState {
    chain.initialize(&state.view(), &boundary);
    let old_analyzers = analyzers;
    let mut analyzers = Vec::new();
    for mut analyzer in old_analyzers {
        analyzer.init(&state.view(), &boundary, &chain, &analyzers);
        analyzers.push(analyzer);
    }

    for ending_criterion in ending_criteria.iter_mut() {
        ending_criterion.initialize(&state.view(), &chain, &boundary);
    }

    let mut rates = Vec::with_capacity(chain.num_possible_reactions(&state));
    for i in 0..chain.num_possible_reactions(&state) {
        rates.push(chain.rate(&state.view(), &boundary, i));
    }
    let rates = BinarySumTree::new(rates);

    let mut reactions_depending_on_locations: Array2<Vec<usize>> = Array2::default(state.raw_dim());
    for i in 0..chain.num_possible_reactions(&state) {
        for indicies in chain.indicies_affecting_reaction(&state.view(), &boundary, i) {
            reactions_depending_on_locations[indicies].push(i);
        }
    }
    let reactions = Vec::new();
    let delta_times = Vec::new();

    SimulationState {
        state,
        boundary,
        chain,
        analyzers,
        ending_criteria,
        rates,
        reactions_depending_on_locations,
        reactions,
        delta_times,
    }
}

fn simulate_iteration(sim_state: &mut SimulationState, rng: &mut impl Rng) -> bool {
    let dt = -(rng.random::<f64>()).ln() / sim_state.rates.sum();
    let chosen_partial_sum = rng.random::<f64>() * sim_state.rates.sum();
    let reaction_id = sim_state.rates.search(chosen_partial_sum);
    let reaction =
        sim_state
            .chain
            .reaction(&sim_state.state.view(), &sim_state.boundary, reaction_id);

    sim_state.reactions.push(reaction);
    sim_state.delta_times.push(dt);
    reaction.apply(&mut sim_state.state);

    sim_state.chain.on_reaction(
        &sim_state.state.view(),
        &sim_state.boundary,
        reaction_id,
        dt,
    );

    let old_analyzers = std::mem::replace(&mut sim_state.analyzers, Vec::new());
    for mut analyzer in old_analyzers {
        analyzer.update(
            &sim_state.state.view(),
            &sim_state.boundary,
            &sim_state.chain,
            reaction,
            dt,
            &sim_state.analyzers,
        );
        sim_state.analyzers.push(analyzer);
    }

    for ending_criterion in sim_state.ending_criteria.iter_mut() {
        ending_criterion.update(
            &sim_state.state.view(),
            &sim_state.chain,
            &sim_state.boundary,
            dt,
            reaction.clone(),
        );
    }

    for ending_criterion in sim_state.ending_criteria.iter() {
        if ending_criterion.should_end() {
            return true; // Signal that simulation should end
        }
    }

    let ids_updated: Vec<usize> = reaction
        .indicies_updated()
        .iter()
        .map(|location| sim_state.reactions_depending_on_locations[*location].iter())
        .fold(Vec::new(), |mut acc, iter| {
            acc.extend(iter);
            acc
        });
    let mut new_rates = Vec::with_capacity(ids_updated.len());
    for &reaction_id in ids_updated.iter() {
        new_rates.push(sim_state.chain.rate(
            &sim_state.state.view(),
            &sim_state.boundary,
            reaction_id,
        ));
    }
    sim_state.rates.batch_update(&ids_updated, &new_rates);

    false // Continue simulation
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn old_simulate(
        mut state: Array2<u32>,
        boundary: Box<dyn BoundaryCondition>,
        mut chain: Box<dyn MarkovChain>,
        mut ending_criteria: Vec<Box<dyn EndingCriterion>>,
        mut rng: impl Rng,
    ) -> (Array2<u32>, Vec<(f64, BasicReaction<u32>)>) {
        chain.initialize(&state.view(), &boundary);
        for ending_criterion in ending_criteria.iter_mut() {
            ending_criterion.initialize(&state.view(), &chain, &boundary);
        }

        let mut rates = Vec::with_capacity(chain.num_possible_reactions(&state));
        for i in 0..chain.num_possible_reactions(&state) {
            rates.push(chain.rate(&state.view(), &boundary, i));
        }
        let mut rates = BinarySumTree::new(rates);

        let mut reactions_depending_on_locations: Array2<Vec<usize>> =
            Array2::default(state.raw_dim());
        for i in 0..chain.num_possible_reactions(&state) {
            for indicies in chain.indicies_affecting_reaction(&state.view(), &boundary, i) {
                reactions_depending_on_locations[indicies].push(i);
            }
        }
        let mut reactions = Vec::new();

        'outer: loop {
            let dt = -(rng.random::<f64>()).ln() / rates.sum();
            let chosen_partial_sum = rng.random::<f64>() * rates.sum();
            let reaction_id = rates.search(chosen_partial_sum);
            let reaction = chain.reaction(&state.view(), &boundary, reaction_id);

            reactions.push((dt, reaction.clone()));
            reaction.apply(&mut state);

            chain.on_reaction(&state.view(), &boundary, reaction_id, dt);

            for ending_criterion in ending_criteria.iter_mut() {
                ending_criterion.update(&state.view(), &chain, &boundary, dt, reaction.clone());
            }

            for ending_criterion in ending_criteria.iter() {
                if ending_criterion.should_end() {
                    break 'outer;
                }
            }

            for location in reaction.indicies_updated() {
                for &reaction_id in reactions_depending_on_locations[location].iter() {
                    rates.update(
                        reaction_id,
                        chain.rate(&state.view(), &boundary, reaction_id),
                    );
                }
            }
        }
        (state, reactions)
    }

    #[test]
    fn simulate_matches_old_simulate() {
        let (width, height) = (50, 50);

        let threshold = 10_000;

        let bond_energy = -3.0;
        let inert_fugacity = 1.0;
        let inert_bonding_rate = 0.1;

        let chemical_potential = 0.;
        let bonding_fugacity = 0.005013 * 6.0;

        let chain = crate::markov_chain::HomogenousChain::new(
            1.0,
            bond_energy,
            chemical_potential,
            inert_fugacity,
            bonding_fugacity,
            inert_bonding_rate,
        );

        let state: Array2<u32> = Array2::default((width, height));
        let ending_criterion = crate::ending_criterion::ReactionCount::new(threshold);
        let boundary = crate::boundary_condition::Periodic;

        let seed = rand::rng().random::<u64>();
        println!("Random seed: {}", seed);
        let rng = StdRng::seed_from_u64(seed);

        let (final_state1, delta_times1, reactions1, _analyzers) = simulate(
            state.clone(),
            Box::new(boundary),
            Box::new(chain.clone()),
            vec![],
            vec![Box::new(ending_criterion)],
            rng.clone(),
        );
        let (final_state2, reactions2) = old_simulate(
            state.clone(),
            Box::new(boundary),
            Box::new(chain),
            vec![Box::new(ending_criterion)],
            rng,
        );
        assert_eq!(final_state1, final_state2);
        assert_eq!(
            reactions1,
            reactions2
                .iter()
                .map(|(_dt, r)| *r)
                .collect::<Vec<BasicReaction<u32>>>()
        );
        assert_eq!(
            delta_times1,
            reactions2.iter().map(|(dt, _r)| *dt).collect::<Vec<f64>>()
        );
    }
}
