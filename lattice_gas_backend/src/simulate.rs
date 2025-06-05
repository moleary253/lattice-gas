use crate::binary_sum_tree::BinarySumTree;
use crate::boundary_condition::BoundaryCondition;
use crate::ending_criterion::EndingCriterion;
use crate::markov_chain::MarkovChain;
use crate::reaction::Reaction;
use numpy::ndarray::Array2;
use rand::prelude::*;

use crate::serialize;
use flate2::{write::GzEncoder, Compression};
use std::fs;

use numpy::prelude::*;
use numpy::PyArray2;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "simulate")]
pub fn py_simulate(
    initial_state: &Bound<'_, PyArray2<u32>>,
    boundary: &Bound<'_, PyAny>,
    chain: &Bound<'_, PyAny>,
    ending_criteria: Vec<Bound<'_, PyAny>>,
    seed: u64,
    output_file: &str,
    save_reactions: bool,
) -> PyResult<()> {
    let state = initial_state.to_owned_array();
    let boundary = crate::boundary_condition::extract(boundary)?;
    let chain = crate::markov_chain::extract(chain)?;

    let mut rust_ending_criteria = Vec::with_capacity(ending_criteria.len());
    for ending_criterion in ending_criteria.iter() {
        rust_ending_criteria.push(crate::ending_criterion::extract(ending_criterion)?);
    }
    let ending_criteria = rust_ending_criteria;

    let rng = StdRng::seed_from_u64(seed);

    let outfile = fs::File::create(output_file)?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);
    if let Err(_) = serialize::serialize_object(
        "initial_conditions.json".to_string(),
        &state,
        &mut archive_builder,
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            "IO error when writing final state to file system.",
        ));
    }

    let (final_state, reactions) = simulate(state, boundary, chain, ending_criteria, rng);

    if let Err(_) = serialize::serialize_object(
        "final_state.json".to_string(),
        &final_state,
        &mut archive_builder,
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            "IO error when writing final state to file system.",
        ));
    }
    if let Err(_) = serialize::serialize_object(
        "final_time.json".to_string(),
        &reactions.iter().map(|(dt, _)| dt).sum::<f64>(),
        &mut archive_builder,
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
            "IO error when writing final state to file system.",
        ));
    }
    if save_reactions {
        if let Err(_) = serialize::serialize_object(
            "reactions.json".to_string(),
            &reactions,
            &mut archive_builder,
        ) {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                "IO error when writing final state to file system.",
            ));
        }
    }

    Ok(())
}

pub fn simulate<T: Clone, R: Reaction<T> + Clone>(
    mut state: Array2<T>,
    boundary: Box<dyn BoundaryCondition<T>>,
    mut chain: Box<dyn MarkovChain<T, R>>,
    mut ending_criteria: Vec<Box<dyn EndingCriterion<T, R> + Send>>,
    mut rng: impl Rng,
) -> (Array2<T>, Vec<(f64, R)>) {
    chain.initialize(&state.view(), &boundary);
    for ending_criterion in ending_criteria.iter_mut() {
        ending_criterion.initialize(&state.view(), &chain, &boundary);
    }

    let mut rates = Vec::with_capacity(chain.num_possible_reactions(&state));
    for i in 0..chain.num_possible_reactions(&state) {
        rates.push(chain.rate(&state.view(), &boundary, i));
    }
    let mut rates = BinarySumTree::new(rates);

    let mut reactions_depending_on_locations: Array2<Vec<usize>> = Array2::default(state.raw_dim());
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
