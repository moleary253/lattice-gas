use lattice_gas::simulate::simulate;
use lattice_gas::*;
use rand::prelude::*;

#[test]
fn homogenous_chain_consistent() {
    let bond_energy = -(1.0_f64).ln();
    let inert_fugacity = 1.0;
    let bonding_fugacity = 2.0;
    let inert_bonding_rate = 0.5;
    let chemical_potential = 1.0_f64.ln();

    let until_step = 100;

    let (width, height) = (20, 20);
    let mut old_system = lattice_gas::System::empty(width, height);
    let old_chain = lattice_gas::HomogenousChain::beta_1(
        bond_energy,
        chemical_potential,
        inert_fugacity,
        bonding_fugacity,
        inert_bonding_rate,
    );

    let state = old_system.state.clone();
    let chain = markov_chain::HomogenousChain::new(
        1.0,
        bond_energy,
        chemical_potential,
        inert_fugacity,
        bonding_fugacity,
        inert_bonding_rate,
    );
    let ending_criterion = ending_criterion::ReactionCount::new(until_step);
    let boundary_condition = boundary_condition::Periodic;

    let mut rng = rand::rngs::StdRng::seed_from_u64(1);

    let (_final_state, reactions) = simulate(
        state,
        boundary_condition,
        chain,
        ending_criterion,
        rng.clone(),
    );

    let mut step = 0;
    loop {
        if step >= until_step {
            break;
        }

        let (dt, reaction) = old_system.next_reaction(&old_chain, &mut rng);
        let difference = (dt - reactions[step].0).abs();
        if difference >= 2_f64.powi(-16) * dt {
            panic!(
                "Difference was too large on step {}: {} vs {}. \nReactions were {:?} and {:?}.",
                step,
                difference,
                1e-16 * dt,
                reaction,
                reactions[step].1
            );
        }

        old_system.update(dt, reaction);

        step += 1;
    }
}
