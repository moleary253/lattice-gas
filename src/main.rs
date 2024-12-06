use flate2::{write::GzEncoder, Compression};
use lattice_gas::serialize;
use lattice_gas::*;
use ndarray::{s, Array2};
use std::fs;
use tqdm::tqdm;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    coexistence()
}

fn coexistence() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (10, 4);

    let beta = 1.0;
    let bond_energy = -2.95;
    let center_driving_chemical_potential = 1.87;
    let driving_chemical_potentials =
        (-3..4).map(|i| i as f64 / 25. + center_driving_chemical_potential);
    let inert_mu = -3.150;
    let bonding_mu = -4.826;
    let inert_to_bonding_rate = 0.1;

    let outfile = fs::File::create("data/coexistence.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let chain = HomogenousChain::new(
        beta,
        bond_energy,
        center_driving_chemical_potential,
        (inert_mu * beta).exp(),
        (bonding_mu * beta).exp(),
        inert_to_bonding_rate,
    );

    let until_step = 10_000;
    let until_time = f64::INFINITY;
    let averaging_time = f64::INFINITY;
    let mut gas_concentration = 0.;
    let mut solid_concentration = 1.;
    let phase_concentration_category = "phase_concentration";
    for (name, state, concentration) in [
        ("gas", SiteState::Empty, &mut gas_concentration),
        ("solid", SiteState::Bonding, &mut solid_concentration),
    ] {
        let mut system = System::full(width, height, Box::new(chain.clone()), state);

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

        let mut step = 0;
        let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
        let mut running_sum = 0.;
        let mut time_measured = 0.;
        loop {
            if step >= until_step || system.time > until_time {
                break;
            }

            let (dt, reaction) = system.next_reaction();
            reactions.push((dt, reaction));

            update_running_average(
                &system,
                dt,
                averaging_time,
                &mut running_sum,
                &mut time_measured,
                &|system: &System| {
                    1. - *(system
                        .particle_number()
                        .get(&SiteState::Empty)
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
    }
    println!("Gas: {}", gas_concentration);
    println!("Solid: {}", solid_concentration);

    let mut initial_state = Array2::default((height, width));
    initial_state
        .slice_mut(s![.., 0..width / 2])
        .fill(SiteState::Bonding);
    let num_trials = 100;
    let coexistence_category = "coexistence";
    for driving_chemical_potential in tqdm(driving_chemical_potentials) {
        for i in tqdm(0..num_trials) {
            let sim_dir = format!(
                "{}/delta_mu={:.2}_trial_{}",
                coexistence_category,
                driving_chemical_potential,
                i + 1
            );

            let chain = HomogenousChain::new(
                beta,
                bond_energy,
                driving_chemical_potential,
                (inert_mu * beta).exp(),
                (bonding_mu * beta).exp(),
                inert_to_bonding_rate,
            );
            serialize::serialize_object(
                format!("{}/chain.json", &sim_dir),
                &chain,
                &mut archive_builder,
            )?;

            let mut system = System::with_state(Box::new(chain), initial_state.clone());
            serialize::serialize_object(
                format!("{}/initial_conditions.json", &sim_dir),
                &system.state,
                &mut archive_builder,
            )?;

            let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
            loop {
                let concentration = 1.
                    - (*(system
                        .particle_number()
                        .get(&SiteState::Empty)
                        .unwrap_or(&0)) as f64
                        / system.state.len() as f64);

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
                    break;
                }

                let (dt, reaction) = system.next_reaction();
                reactions.push((dt, reaction));

                system.update(dt, reaction);
            }
            serialize::serialize_object(
                format!("{}/reactions.json", &sim_dir),
                &reactions,
                &mut archive_builder,
            )?;
        }
    }

    archive_builder.finish()?;

    Ok(())
}

fn ising_test() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (50, 50);

    let betas = (0..10).map(|x| x as f64 / 50. + 0.36);
    let bond_energy = -4.0;
    let driving_chemical_potential = 0.0;
    let inert_fugacity = 0.;
    let bonding_mu = 2. * bond_energy;
    let inert_to_bonding_rate = 0.0;

    let until_time = 10_000.;
    let until_step = 1_000_000;
    let averaging_time = 1000.;

    for beta in betas {
        let dir = format!("data/test/beta={:.2}", beta);
        fs::create_dir_all(&dir)?;

        let chain = HomogenousChain::new(
            beta,
            bond_energy,
            driving_chemical_potential,
            inert_fugacity,
            (bonding_mu * beta).exp(),
            inert_to_bonding_rate,
        );
        fs::write(
            format!("{}/chain.json", &dir),
            serde_json::to_string(&chain)?,
        )?;

        let mut system = System::full(width, height, Box::new(chain), SiteState::Bonding);
        fs::write(
            format!("{}/initial_conditions.json", &dir),
            serde_json::to_string(&system.state)?,
        )?;

        let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();

        let mut running_sum = 0.;
        let mut time_measured = 0.;

        println!("\nRun with beta = {}", beta);
        let mut i = 0;
        loop {
            i += 1;
            if system.time > until_time || i == until_step {
                break;
            }

            let (dt, reaction) = system.next_reaction();
            reactions.push((dt, reaction));

            match update_running_average(
                &system,
                dt,
                averaging_time,
                &mut running_sum,
                &mut time_measured,
                &|system: &System| {
                    *(system
                        .particle_number()
                        .get(&SiteState::Bonding)
                        .unwrap_or(&0)) as f64
                        / system.state.len() as f64
                },
            ) {
                None => {}
                Some(average) => {
                    println!("Average at {:.0} sec: {}", system.time.round(), average);
                }
            }

            system.update(dt, reaction);
        }

        fs::write(
            format!("{}/reactions.json", &dir),
            serde_json::to_string(&reactions)?,
        )?;

        println!(
            "Final average at {:.2} sec: {}",
            system.time,
            running_sum / time_measured
        );
    }
    Ok(())
}
