use boundary_condition::BoundaryCondition;
use lattice_gas::simulate::simulate;
use lattice_gas::*;
use ndarray::Array2;
use rand::prelude::*;
use std::any::Any;

#[test]
fn droplet_analysis_consistent_over_time() {
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
    let boundary = boundary_condition::Periodic;

    let is_droplet = vec![markov_chain::BONDING];
    let droplets = analysis::Droplets::new(
        &state.view(),
        &(Box::new(boundary) as Box<dyn BoundaryCondition>),
        &is_droplet,
    );

    let ending_criterion = ending_criterion::ReactionCount::new(threshold);

    let seed = rand::rng().random::<u64>();
    println!("Random seed: {}", seed);
    let rng = StdRng::seed_from_u64(seed);

    let (final_state, _delta_times, _reactions, mut analyzers) = simulate(
        state,
        Box::new(boundary),
        Box::new(chain),
        vec![Box::new(droplets)],
        vec![Box::new(ending_criterion)],
        rng,
    );

    let droplets = analyzers.remove(0);
    let Ok(droplets) = (droplets as Box<dyn Any>).downcast::<analysis::Droplets>() else {
        panic!("Cast didn't work");
    };

    let expected_droplets = analysis::Droplets::new(
        &final_state.view(),
        &(Box::new(boundary) as Box<dyn BoundaryCondition>),
        &is_droplet,
    );

    let mut label_translator = std::collections::HashMap::new();
    for (pos, &label) in droplets.labeled.indexed_iter() {
        let expected_label = expected_droplets.labeled[pos];
        if label == 0 {
            assert_eq!(expected_label, 0);
            continue;
        }
        if let Some(translated_label) = label_translator.get(&label) {
            assert_eq!(expected_label, *translated_label);
            continue;
        }
        println!(
            "Establishing {} = {} at {:?} with value {:?}",
            label, expected_label, pos, final_state[pos]
        );
        assert!(expected_label != 0);
        label_translator.insert(label, expected_label);
        assert_eq!(
            droplets.droplets[label - 1].len(),
            expected_droplets.droplets[expected_label - 1].len()
        );
        for index in droplets.droplets[label - 1].iter() {
            assert!(expected_droplets.droplets[expected_label - 1].contains(index));
        }
    }
}
