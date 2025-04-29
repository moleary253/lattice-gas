use crate::binary_sum_tree::BinarySumTree;
use crate::boundary_condition::BoundaryCondition;
use crate::ending_criterion::EndingCriterion;
use crate::markov_chain::MarkovChain;
use crate::reaction::Reaction;
use ndarray::Array2;
use rand::prelude::*;

pub fn simulate<T: Clone, R: Reaction<T> + Clone>(
    mut state: Array2<T>,
    boundary: impl BoundaryCondition<T>,
    mut chain: impl MarkovChain<T, R>,
    mut ending_criterion: impl EndingCriterion<T, R>,
    mut rng: impl Rng,
) -> (Array2<T>, Vec<(f64, R)>) {
    let mut rates = Vec::with_capacity(chain.num_possible_reactions(&state));
    for i in 0..chain.num_possible_reactions(&state) {
        rates.push(chain.rate(&state, &boundary, i));
    }
    let mut rates = BinarySumTree::new(rates);

    let mut reactions_depending_on_locations: Array2<Vec<usize>> = Array2::default(state.raw_dim());
    for i in 0..chain.num_possible_reactions(&state) {
        for indicies in chain.indicies_affecting_reaction(&state, &boundary, i) {
            reactions_depending_on_locations[indicies].push(i);
        }
    }

    ending_criterion.initialize(&state, &chain);
    chain.initialize(&state, &boundary);
    let mut reactions = Vec::new();

    loop {
        let dt = -(rng.random::<f64>()).ln() / rates.sum();
        let chosen_partial_sum = rng.random::<f64>() * rates.sum();
        let reaction_id = rates.search(chosen_partial_sum);
        let reaction = chain.reaction(&state, reaction_id);
        reactions.push((dt, reaction.clone()));

        ending_criterion.update(&state, &chain, dt, reaction.clone());
        if ending_criterion.should_end() {
            break;
        }

        chain.on_reaction(&state, &boundary, reaction_id, dt);

        reaction.apply(&mut state);

        for location in reaction.indicies_updated(&state) {
            for &reaction_id in reactions_depending_on_locations[location].iter() {
                rates.update(reaction_id, chain.rate(&state, &boundary, reaction_id));
            }
        }
    }
    (state, reactions)
}
