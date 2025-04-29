use flate2::{write::GzEncoder, Compression};
use lattice_gas::serialize;
use lattice_gas::*;
use ndarray::{s, Array2};
use std::fs;
// use std::thread;
use tqdm::tqdm;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    profile_new()
}

fn profile_new() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (100, 100);
    let threshold = 10_000;

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
    let category = "propensity";
    let ending_criterion = crate::ending_criterion::ReactionCount::new(threshold);
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

            let (final_state, reactions) = crate::simulate::simulate(
                state,
                boundary_condition::Periodic,
                chain,
                ending_criterion,
                rand::rng(),
            );
            // serialize::serialize_object(
            //     format!("{}/reactions.json", &sim_dir),
            //     &reactions,
            //     &mut archive_builder,
            // )?;
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

fn profile_old() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (19, 19);
    let threshold = 10_000;

    let bond_energy = -3.0;
    let inert_fugacity = 1.0;
    let inert_bonding_rate = 0.1;

    let chemical_potential = 0.;
    let base_bonding_fugacity = 0.005013;
    let bonding_fugacity_multipliers = [2.5, 3.0, 6.0];

    let outfile = fs::File::create("data/profile_old.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let num_trials = 100;
    let category = "propensity";
    let mut rng = rand::rng();
    for bonding_fugacity_multiplier in bonding_fugacity_multipliers {
        let bonding_fugacity = base_bonding_fugacity * bonding_fugacity_multiplier;

        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/mult={:.2}_trial_{}",
                category,
                bonding_fugacity_multiplier,
                i + 1
            );

            let chain = HomogenousChain::beta_1(
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

            let mut system = System::empty(width, height);
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;

            let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
            let mut step = 0;
            loop {
                if step >= threshold {
                    break;
                }

                let (dt, reaction) = system.next_reaction(&chain, &mut rng);
                reactions.push((dt, reaction));

                system.update(dt, reaction);
                step += 1;
            }
            serialize::serialize_object(
                format!("{}/reactions.json", &sim_dir),
                &reactions,
                &mut archive_builder,
            )?;
            serialize::serialize_object(
                format!("{}/final_state.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}

fn experiment3() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (19, 19);

    let bond_energy = -3.0;
    let inert_fugacity = 1.0;
    let inert_bonding_rate = 0.1;

    let chemical_potential = 0.;
    let base_bonding_fugacity = 0.005013;
    let bonding_fugacity_multipliers = [2.5, 3.0, 6.0];

    let outfile = fs::File::create("data/experiment3.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let mut particle_number = ParticleNumberStatistic::new();

    let num_trials = 100;
    let threshold = 0.1;
    let category = "propensity";
    let mut rng = rand::rng();
    for bonding_fugacity_multiplier in bonding_fugacity_multipliers {
        let bonding_fugacity = base_bonding_fugacity * bonding_fugacity_multiplier;

        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/mult={:.2}_trial_{}",
                category,
                bonding_fugacity_multiplier,
                i + 1
            );

            let chain = HomogenousChain::beta_1(
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

            let mut system = System::empty(width, height);
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;

            particle_number.initialize(&system);

            let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
            loop {
                let concentration = *(particle_number
                    .value()
                    .get(&SiteState::Bonding)
                    .unwrap_or(&0)) as f64
                    / system.state.len() as f64;
                // println!("{:.3}", concentration);
                if concentration >= threshold {
                    serialize::serialize_object(
                        format!("{}/outcome.txt", &sim_dir),
                        &format!("{}", system.time),
                        &mut archive_builder,
                    )?;
                    break;
                }

                let (dt, reaction) = system.next_reaction(&chain, &mut rng);
                reactions.push((dt, reaction));

                particle_number.update(&system, dt, reaction);

                system.update(dt, reaction);
            }
            serialize::serialize_object(
                format!("{}/reactions.json", &sim_dir),
                &reactions,
                &mut archive_builder,
            )?;
            serialize::serialize_object(
                format!("{}/final_state.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}

fn experiment2() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut rng = rand::rng();
    let (width, height) = (19, 19);

    let bond_energy = -3.0;
    let inert_fugacity = 1.0;
    let inert_bonding_rate = 0.1;

    let chemical_potentials = [-2., -1., 0., 1., 2.];
    let bonding_fugacities = [0.000702, 0.001884, 0.005013, 0.012695, 0.029053];
    let bonding_fugacity_multiplier = 3.0;

    let outfile = fs::File::create("data/experiment2.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let mut particle_number = ParticleNumberStatistic::new();

    let propensity_category = "propensity";

    let num_trials = 100;
    let threshold = 0.1;
    for (chemical_potential, bonding_fugacity) in
        chemical_potentials.iter().zip(bonding_fugacities.iter())
    {
        let bonding_fugacity = bonding_fugacity * bonding_fugacity_multiplier;
        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/mu={:.2}_trial_{}",
                propensity_category,
                chemical_potential,
                i + 1
            );

            let chain = HomogenousChain::beta_1(
                bond_energy,
                *chemical_potential,
                inert_fugacity,
                bonding_fugacity,
                inert_bonding_rate,
            );
            serialize::serialize_object(
                format!("{}/chain.json", &sim_dir),
                &chain,
                &mut archive_builder,
            )?;

            let mut system = System::empty(width, height);
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;

            particle_number.initialize(&system);

            let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
            loop {
                let concentration = *(particle_number
                    .value()
                    .get(&SiteState::Bonding)
                    .unwrap_or(&0)) as f64
                    / system.state.len() as f64;
                // println!("{:.3}", concentration);
                if concentration >= threshold {
                    serialize::serialize_object(
                        format!("{}/outcome.txt", &sim_dir),
                        &format!("{}", system.time),
                        &mut archive_builder,
                    )?;
                    break;
                }

                let (dt, reaction) = system.next_reaction(&chain, &mut rng);
                reactions.push((dt, reaction));

                particle_number.update(&system, dt, reaction);

                system.update(dt, reaction);
            }
            serialize::serialize_object(
                format!("{}/reactions.json", &sim_dir),
                &reactions,
                &mut archive_builder,
            )?;
            serialize::serialize_object(
                format!("{}/final_state.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}

fn experiment1() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut rng = rand::rng();
    let (width, height) = (19, 19);

    let bond_energy = -3.0;
    let inert_fugacity = 1.0;
    let inert_bonding_rate = 0.1;

    let num_chemical_potentials = 5;
    let (min_chemical_potential, max_chemical_potential) = (-2.0, 2.0);
    let chemical_potentials = (0..num_chemical_potentials).map(|i| {
        i as f64 / (num_chemical_potentials - 1) as f64
            * (-min_chemical_potential + max_chemical_potential)
            + min_chemical_potential
    });

    let outfile = fs::File::create("data/experiment1.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let mut particle_number = ParticleNumberStatistic::new();

    let chain = HomogenousChain::beta_1(bond_energy, 0., inert_fugacity, 0.01, inert_bonding_rate);

    let until_step = 10_000;
    let until_time = f64::INFINITY;
    let averaging_time = f64::INFINITY;
    let mut gas_concentration = 0.;
    let mut solid_concentration = 1.;
    let phase_concentration_category = "phase_concentrations";
    for (name, state, concentration) in [
        ("gas", SiteState::Inert, &mut gas_concentration),
        ("solid", SiteState::Bonding, &mut solid_concentration),
    ] {
        let mut system = System::full(width, height, state);

        serialize::serialize_object(
            format!("{}/{}/chain.json", &phase_concentration_category, &name),
            &chain,
            &mut archive_builder,
        )?;
        serialize::serialize_object(
            format!(
                "{}/{}/initial_conditions.json",
                &phase_concentration_category, &name
            ),
            &system.state,
            &mut archive_builder,
        )?;

        particle_number.initialize(&system);
        let mut step = 0;
        let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
        let mut running_sum = 0.;
        let mut time_measured = 0.;
        loop {
            if step >= until_step || system.time > until_time {
                break;
            }

            let (dt, reaction) = system.next_reaction(&chain, &mut rng);

            reactions.push((dt, reaction));

            particle_number.update(&system, dt, reaction);
            update_running_average(
                &system,
                dt,
                averaging_time,
                &mut running_sum,
                &mut time_measured,
                &|system: &System| {
                    *(particle_number
                        .value()
                        .get(&SiteState::Bonding)
                        .unwrap_or(&0)) as f64
                        / system.state.len() as f64
                },
            );

            system.update(dt, reaction);

            step += 1;
        }

        *concentration = running_sum / time_measured;
        serialize::serialize_object(
            format!("{}/{}/reactions.json", &phase_concentration_category, &name),
            &reactions,
            &mut archive_builder,
        )?;
        serialize::serialize_object(
            format!(
                "{}/{}/final_state.json",
                &phase_concentration_category, &name
            ),
            &system.state,
            &mut archive_builder,
        )?;
    }
    println!("Gas: {}", gas_concentration);
    println!("Solid: {}", solid_concentration);

    // mu = -2 -> z_B = 0.000702
    // mu = -1 -> z_B = 0.001884
    // mu =  0 -> z_B = 0.005013
    // mu =  1 -> z_B = 0.012695
    // mu =  2 -> z_B = 0.029053
    let mut initial_state = Array2::default((height, width));
    initial_state
        .slice_mut(s![.., 0..width / 2])
        .fill(SiteState::Bonding);
    let trials_per_step = 10;
    let tolerance = (trials_per_step as f64).sqrt() / 2.0;
    let coexistence_category = "coexistence";
    for chemical_potential in chemical_potentials {
        let mut fugacity_search_bounds = [0.0, 1.0];
        loop {
            let mut solid_count = 0;
            let fugacity = fugacity_search_bounds.iter().sum::<f64>() / 2.;

            let chain = HomogenousChain::beta_1(
                bond_energy,
                chemical_potential,
                inert_fugacity,
                fugacity,
                inert_bonding_rate,
            );

            println!();
            for i in tqdm(0..trials_per_step) {
                let sim_dir = format!(
                    "{}/delta_mu={:.2}_z_B={:.2}_trial_{}",
                    coexistence_category,
                    chemical_potential,
                    fugacity,
                    i + 1
                );

                serialize::serialize_object(
                    format!("{}/chain.json", &sim_dir),
                    &chain,
                    &mut archive_builder,
                )?;
                let mut system = System::with_state(initial_state.clone());
                serialize::serialize_object(
                    format!("{}/initial_conditions.json", &sim_dir),
                    &system.state,
                    &mut archive_builder,
                )?;

                particle_number.initialize(&system);
                let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
                loop {
                    let concentration = *(particle_number
                        .value()
                        .get(&SiteState::Bonding)
                        .unwrap_or(&0)) as f64
                        / system.state.len() as f64;

                    if concentration <= gas_concentration {
                        serialize::serialize_object(
                            format!("{}/outcome.txt", &sim_dir),
                            &"gas",
                            &mut archive_builder,
                        )?;
                        break;
                    }
                    if concentration >= solid_concentration {
                        serialize::serialize_object(
                            format!("{}/outcome.txt", &sim_dir),
                            &"solid",
                            &mut archive_builder,
                        )?;
                        solid_count += 1;
                        break;
                    }

                    let (dt, reaction) = system.next_reaction(&chain, &mut rng);

                    particle_number.update(&system, dt, reaction);
                    reactions.push((dt, reaction));

                    system.update(dt, reaction);
                }
                serialize::serialize_object(
                    format!("{}/reactions.json", &sim_dir),
                    &reactions,
                    &mut archive_builder,
                )?;
                serialize::serialize_object(
                    format!("{}/final_state.json", &sim_dir),
                    &system.state,
                    &mut archive_builder,
                )?;
            }

            println!(
                "Mu = {:.2}, Fugacity = {:.6}, Num Solid = {:.2}",
                chemical_potential, fugacity, solid_count
            );
            let error = solid_count - trials_per_step / 2;
            if (error as f64).abs() < tolerance {
                break;
            } else if error < 0 {
                fugacity_search_bounds[0] = fugacity;
            } else {
                fugacity_search_bounds[1] = fugacity;
            }
        }
    }

    archive_builder.finish()?;

    Ok(())
}

fn propensity() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let mut rng = rand::rng();
    let (width, height) = (19, 19);

    let beta = 1.0;
    let bond_energy = -2.95;

    let num_chemical_potentials = 5;
    let (min_chemical_potential, max_chemical_potential) = (0., 4.0);
    let kinase_chemical_potentials = (0..num_chemical_potentials).map(|i| {
        i as f64 / (num_chemical_potentials - 1) as f64
            * (-min_chemical_potential + max_chemical_potential)
            + min_chemical_potential
    });
    let kinase_michaelis_constant = 10.;
    let kinase_rate_constant = 0.1 * kinase_michaelis_constant;

    let zeta_chain_extension_stdev = 4.;
    let t_cell_receptor_position = [9, 9];

    let phosphatase_chemical_potential = 0.0;
    let phosphatase_michaelis_constant = 1.;
    let phosphatase_rate_constant = 0.0 * phosphatase_michaelis_constant;

    let outfile = fs::File::create("data/propensity.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let num_trials = 100;
    let threshold = 0.08;
    let propensity_category = "propensity";
    let odds = [
        (SiteState::Empty, 10.),
        (SiteState::Inert, 2.),
        (SiteState::Bonding, 0.0),
    ];

    let mut particle_number = ParticleNumberStatistic::new();

    for kinase_chemical_potential in tqdm(kinase_chemical_potentials) {
        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/delta_mu={:.2}_trial_{}",
                propensity_category,
                kinase_chemical_potential,
                i + 1
            );

            let mut chain = TCellChain::new(
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
            );
            serialize::serialize_object(
                format!("{}/chain.json", &sim_dir),
                &chain,
                &mut archive_builder,
            )?;

            let mut system = System::random(width, height, odds);
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;

            particle_number.initialize(&system);
            chain.initialize_cache(&system);

            let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
            loop {
                let concentration = *(particle_number
                    .value()
                    .get(&SiteState::Bonding)
                    .unwrap_or(&0)) as f64
                    / system.state.len() as f64;
                if concentration >= threshold {
                    serialize::serialize_object(
                        format!("{}/outcome.txt", &sim_dir),
                        &format!("{}", system.time),
                        &mut archive_builder,
                    )?;
                    break;
                }

                let (dt, reaction) = system.next_reaction(&chain, &mut rng);
                reactions.push((dt, reaction));

                chain.update_cache(&system, dt, reaction);
                particle_number.update(&system, dt, reaction);

                system.update(dt, reaction);
            }
            serialize::serialize_object(
                format!("{}/reactions.json", &sim_dir),
                &reactions,
                &mut archive_builder,
            )?;
            serialize::serialize_object(
                format!("{}/final_state.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}
