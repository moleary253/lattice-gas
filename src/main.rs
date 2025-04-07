use flate2::{write::GzEncoder, Compression};
use lattice_gas::serialize;
use lattice_gas::*;
use std::fs;
use tqdm::tqdm;

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    propensity()
}

fn profile() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (19, 19);

    let beta = 1.0;
    let bond_energy = -2.95;

    let kinase_chemical_potential = 1.2;
    let kinase_rate_constant = 0.1;
    let kinase_michaelis_constant = 10.;

    let zeta_chain_extension_stdev = 3.;
    let t_cell_receptor_position = [9, 9];

    let phosphatase_chemical_potential = 0.0;
    let phosphatase_michaelis_constant = 1000.;
    let phosphatase_rate_constant = 0.1 * phosphatase_michaelis_constant;

    let outfile = fs::File::create("data/profile.tar.gz")?;
    let zipper = GzEncoder::new(outfile, Compression::default());
    let mut archive_builder = tar::Builder::new(zipper);

    let num_trials = 10;
    let threshold = 0.08;
    let category = "profile";
    let odds = [
        (SiteState::Empty, 10.),
        (SiteState::Inert, 2.),
        (SiteState::Bonding, 0.0),
    ];

    for i in tqdm(0..num_trials) {
        let sim_dir = format!("{}/trial_{}", category, i + 1);

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

        let mut particle_number = ParticleNumberStatistic::new();
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

            let (dt, reaction) = system.next_reaction(&chain);
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
    }

    archive_builder.finish()?;

    Ok(())
}

fn propensity() -> Result<(), Box<dyn std::error::Error + 'static>> {
    let (width, height) = (19, 19);

    let beta = 1.0;
    let bond_energy = -2.95;

    let num_chemical_potentials = 5;
    let (min_chemical_potential, max_chemical_potential) = (0., 1.2);
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

                let (dt, reaction) = system.next_reaction(&chain);
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
        }
    }

    archive_builder.finish()?;

    Ok(())
}

// fn coexistence() -> Result<(), Box<dyn std::error::Error + 'static>> {
//     let (width, height) = (19, 19);

//     let beta = 1.0;
//     let bond_energy = -2.95;
//     let center_driving_chemical_potential = 1.87;
//     let driving_chemical_potentials =
//         (-3..4).map(|i| i as f64 / 25. + center_driving_chemical_potential);
//     let phosphorylation_rate_constant = 0.1;
//     let zeta_chain_extension_stdev = 3.;
//     let reaction_box_size = 0.;
//     let t_cell_receptor_position = [10, 10];

//     let outfile = fs::File::create("data/coexistence.tar.gz")?;
//     let zipper = GzEncoder::new(outfile, Compression::default());
//     let mut archive_builder = tar::Builder::new(zipper);

//     let chain = TCellChain::new(
//         beta,
//         bond_energy,
//         center_driving_chemical_potential,
//         phosphorylation_rate_constant,
//         zeta_chain_extension_stdev,
//         reaction_box_size,
//         t_cell_receptor_position,
//     );

//     let until_step = 10_000;
//     let until_time = f64::INFINITY;
//     let averaging_time = f64::INFINITY;
//     let mut gas_concentration = 0.;
//     let mut solid_concentration = 1.;
//     let phase_concentration_category = "phase_concentration";
//     for (name, state, concentration) in [
//         ("gas", SiteState::Inert, &mut gas_concentration),
//         ("solid", SiteState::Bonding, &mut solid_concentration),
//     ] {
//         let mut system = System::full(width, height, Box::new(chain.clone()), state);

//         serialize::serialize_object(
//             format!("{}/{}/chain.json", &phase_concentration_category, &name),
//             &chain,
//             &mut archive_builder,
//         )?;
//         serialize::serialize_object(
//             format!(
//                 "{}/{}/initial_conditions.json",
//                 &phase_concentration_category, &name
//             ),
//             &system.state,
//             &mut archive_builder,
//         )?;

//         let mut step = 0;
//         let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
//         let mut running_sum = 0.;
//         let mut time_measured = 0.;
//         loop {
//             if step >= until_step || system.time > until_time {
//                 break;
//             }

//             let (dt, reaction) = system.next_reaction();
//             reactions.push((dt, reaction));

//             update_running_average(
//                 &system,
//                 dt,
//                 averaging_time,
//                 &mut running_sum,
//                 &mut time_measured,
//                 &|system: &System| {
//                     *(system
//                         .particle_number()
//                         .get(&SiteState::Bonding)
//                         .unwrap_or(&0)) as f64
//                         / system.state.len() as f64
//                 },
//             );

//             system.update(dt, reaction);

//             step += 1;
//         }

//         *concentration = running_sum / time_measured;
//         serialize::serialize_object(
//             format!("{}/{}/reactions.json", &phase_concentration_category, &name),
//             &reactions,
//             &mut archive_builder,
//         )?;
//     }
//     println!("Gas: {}", gas_concentration);
//     println!("Solid: {}", solid_concentration);

//     archive_builder.finish()?;

//     return Ok(());

//     let mut initial_state = Array2::default((height, width));
//     initial_state
//         .slice_mut(s![.., 0..width / 2])
//         .fill(SiteState::Bonding);
//     let num_trials = 100;
//     let coexistence_category = "coexistence";
//     for driving_chemical_potential in tqdm(driving_chemical_potentials) {
//         for i in tqdm(0..num_trials) {
//             let sim_dir = format!(
//                 "{}/delta_mu={:.2}_trial_{}",
//                 coexistence_category,
//                 driving_chemical_potential,
//                 i + 1
//             );

//             let chain = TCellChain::new(
//                 beta,
//                 bond_energy,
//                 driving_chemical_potential,
//                 phosphorylation_rate_constant,
//                 zeta_chain_extension_stdev,
//                 reaction_box_size,
//                 t_cell_receptor_position,
//             );
//             serialize::serialize_object(
//                 format!("{}/chain.json", &sim_dir),
//                 &chain,
//                 &mut archive_builder,
//             )?;

//             let mut system = System::with_state(Box::new(chain), initial_state.clone());
//             serialize::serialize_object(
//                 format!("{}/initial_conditions.json", &sim_dir),
//                 &system.state,
//                 &mut archive_builder,
//             )?;

//             let mut reactions: Vec<(f64, Reaction<SiteState>)> = Vec::new();
//             loop {
//                 let concentration = 1.
//                     - (*(system
//                         .particle_number()
//                         .get(&SiteState::Empty)
//                         .unwrap_or(&0)) as f64
//                         / system.state.len() as f64);

//                 if concentration <= gas_concentration {
//                     serialize::serialize_object(
//                         format!("{}/outcome.txt", &sim_dir),
//                         &"gas",
//                         &mut archive_builder,
//                     )?;
//                     break;
//                 }
//                 if concentration >= solid_concentration {
//                     serialize::serialize_object(
//                         format!("{}/outcome.txt", &sim_dir),
//                         &"solid",
//                         &mut archive_builder,
//                     )?;
//                     break;
//                 }

//                 let (dt, reaction) = system.next_reaction();
//                 reactions.push((dt, reaction));

//                 system.update(dt, reaction);
//             }
//             serialize::serialize_object(
//                 format!("{}/reactions.json", &sim_dir),
//                 &reactions,
//                 &mut archive_builder,
//             )?;
//         }
//     }

//     archive_builder.finish()?;

//     Ok(())
// }

fn concentration_near_tcr(system: &System, chain: &TCellChain) -> f64 {
    let mut num_bonding = 0;
    let mut total = 0;
    for (i, state) in system.state.iter().enumerate() {
        let [x, y] = system.pos_of_ith_site(i);
        let (x, y) = (x as f64, y as f64);
        let (dx, dy) = (
            x - chain.t_cell_receptor_position[0] as f64,
            y - chain.t_cell_receptor_position[1] as f64,
        );
        let distance_squared = (dx * dx + dy * dy);
        if distance_squared / chain.zeta_chain_extension_stdev.powi(2) > 1. {
            continue;
        }
        total += 1;
        if *state != SiteState::Bonding {
            continue;
        }
        num_bonding += 1;
    }
    num_bonding as f64 / total as f64
}
