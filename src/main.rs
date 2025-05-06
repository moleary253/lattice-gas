use flate2::{write::GzEncoder, Compression};
use lattice_gas::serialize;
use lattice_gas::*;
use ndarray::Array2;
use rand::prelude::*;
use std::fs;
use tqdm::tqdm;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    profile()
}

fn profile() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (100, 100);

    let bond_energy = -3.0;
    let inert_fugacity = 1.0;
    let inert_bonding_rate = 0.1;

    let chemical_potential = 0.;
    let base_bonding_fugacity = 0.005013;
    let bonding_fugacity_multipliers = [2.5, 3.0, 6.0];

    let outfile = fs::File::create("data/profile_new.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let num_trials = 100;
    let num_to_save_reactions = 2;
    let category = "propensity";
    let mut seed_generator = rand::rng();
    let ending_criterion =
        ending_criterion::LargestDropletSize::new(9000, |site| *site == SiteState::Bonding);
    for bonding_fugacity_multiplier in bonding_fugacity_multipliers {
        let bonding_fugacity = base_bonding_fugacity * bonding_fugacity_multiplier;

        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/mult={:.2}_trial_{}",
                category,
                bonding_fugacity_multiplier,
                i + 1
            );

            let chain = crate::markov_chain::HomogenousChain::new(
                1.0,
                bond_energy,
                chemical_potential,
                inert_fugacity,
                bonding_fugacity,
                inert_bonding_rate,
            );
            serialize::serialize_object(
                format!("{}/chain.json", &sim_dir),
                &chain,
                &mut archive_builder,
            )?;

            let state: Array2<SiteState> = Array2::default((width, height));
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &state,
                &mut archive_builder,
            )?;

            let boundary = boundary_condition::Periodic;
            serialize::serialize_object(
                format!("{}/boundary.json", &sim_dir),
                &boundary,
                &mut archive_builder,
            )?;

            let seed = seed_generator.random::<u64>();
            serialize::serialize_object(
                format!("{}/seed.json", &sim_dir),
                &seed,
                &mut archive_builder,
            )?;

            let (final_state, reactions) = crate::simulate::simulate(
                state.clone(),
                boundary,
                chain,
                ending_criterion.clone(),
                StdRng::seed_from_u64(seed),
            );
            let sizes =
                analysis::largest_droplet_size_over_time(&state, &boundary, &reactions, |site| {
                    *site == SiteState::Bonding
                });
            serialize::serialize_object(
                format!("{}/sizes.json", &sim_dir),
                &sizes,
                &mut archive_builder,
            )?;
            if i < num_to_save_reactions {
                serialize::serialize_object(
                    format!("{}/reactions.json", &sim_dir),
                    &reactions,
                    &mut archive_builder,
                )?;
            }
            serialize::serialize_object(
                format!("{}/final_state.json", &sim_dir),
                &final_state,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}
