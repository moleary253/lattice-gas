import lattice_gas as lg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import load


DEFAULT_DIR="data/effect_of_activity"


def run_simulation(output_file, random_seed, bonding_fugacity, initial_state, ending_criteria):
    boundary = lg.boundary_condition.Periodic()

    chain = lg.markov_chain.HomogenousChain(
        beta=1.0,
        bond_energy=-3.0,
        driving_chemical_potential=0.0,
        inert_fugacity=0.0,
        bonding_fugacity=bonding_fugacity,
        inert_to_bonding_rate=0.0,
    )


    lg.simulate.simulate(
        initial_state,
        boundary,
        chain,
        ending_criteria,
        random_seed,
        output_file,
        save_reactions=True,
    )


def run():
    import shutil

    num_trials = 100
    z_B_coex = 0.0024609375 # z_I = 0
    # z_B_coex = 0.002424609375 # z_I = 0.1
    low_fugacities = [z_B_coex / 3]
    high_fugacities = [z_B_coex * 1.5, z_B_coex * 3]
    
    if os.path.exists(DEFAULT_DIR):
        shutil.rmtree(DEFAULT_DIR)
    os.mkdir(DEFAULT_DIR)
    os.mkdir(os.path.join(DEFAULT_DIR, "low"))
    os.mkdir(os.path.join(DEFAULT_DIR, "high"))
    for i in tqdm(range(num_trials)):
        for low_fugacity in low_fugacities:
            random_seed = 17 + i * 31
            output_file = DEFAULT_DIR + f"/low/{low_fugacity:.5f}_{i+1}.tar.gz"
            ending_criterion = lg.ending_criterion.ReactionCount(100_000)
            initial_state = np.zeros((100, 100), dtype=np.dtype("u4"))
            run_simulation(output_file, random_seed, low_fugacity, initial_state, [ending_criterion])

            directory = load.unpack_natural_input(output_file)
            final_conditions = load.final_state(directory)
            shutil.rmtree(load.TEMP_ARCHIVE_PATH)

            for high_fugacity in high_fugacities:
                output_file = (
                    DEFAULT_DIR
                    + f"/high/{low_fugacity:.5f}_to_{high_fugacity:.5f}_{i+1}.tar.gz"
                )
                random_seed = 19 + i * 37 + int(high_fugacity * 10000) % 11
                ending_criterion = lg.ending_criterion.LargestDropletSize(
                    100,
                    [load.BONDING],
                )
                run_simulation(
                    output_file,
                    random_seed,
                    high_fugacity,
                    final_conditions,
                    [ending_criterion],
                )


def analyze(path):
    from scipy.stats import gaussian_kde
    def is_bonding(x):
        return x == load.BONDING

    boundary = lg.boundary_condition.Periodic()

    times = {}
    threshold = 10
    high_fugacities = {}
    for sim_name in tqdm(os.listdir(os.path.join(path, "high"))):
        sim_path = os.path.join(path, "high", sim_name)
        directory = load.unpack_natural_input(sim_path)

        low_fugacity = float(sim_name.split("_")[0])
        high_fugacity = float(sim_name.split("_")[2])
        delta_mu = np.log(high_fugacity / low_fugacity)
        group = f"$z_B={low_fugacity:.5f}$ to $z_B={high_fugacity:.4f}$"
        if group not in times:
            times[group] = []
            high_fugacities[group] = high_fugacity

        initial_state = load.initial_conditions()

        # state = initial_state.copy().astype("u4")
        # reactions = load.reactions()
        # droplets = lg.analysis.Droplets(
        #     state,
        #     boundary,
        #     [load.BONDING],
        # )
        # t = 0
        # for reaction in reactions:
        #     load.apply_reaction(state, reaction)
        #     droplets.update(
        #         state,
        #         boundary,
        #         [load.BONDING],
        #         reaction,
        #     )
        #     t += reaction["dt"]
        #     if np.max([len(drop) for drop in droplets.droplets()]) == threshold:
        #         break
        # time = t

        time = load.final_time()
        times[group].append(time)
        load.clean_up_temp_files()
        
    # for group in times:
    #     times[group] = np.array(times[group]) / np.average(times[group])

    fig1, ax1s = plt.subplots(len(times), 1)

    upper_time_limit = 2
    bins = np.linspace(0, upper_time_limit, 90)
    sample_times = np.linspace(np.min(bins), np.max(bins), 300)

    pdfs = [gaussian_kde(times[group]) for group in times]

    for i, group in enumerate(times):
        print(f"{group}: sigma = {np.std(times[group]):.2f}")
        color = (
            0.8 * i / (len(times) - 1),
            0.,
            0.8 * (1 - i / (len(times) - 1)),
            0.5
        )
        n, x, _ = ax1s[i].hist(
            times[group],
            bins=bins * np.average(times[group]), 
            histtype=u'bar',
            density=True,
            label=group,
            color=color
        )
        ax1s[i].plot(
            np.average(times[group]) * sample_times,
            pdfs[i](np.average(times[group]) * sample_times),
            color=color
        )
        ax1s[i].plot(
            np.average(times[group]) * sample_times,
            cnt_predicted_pdf(
                np.average(times[group]) * sample_times,
                beta=1.0,
                bond_energy=-3.0,
                bonding_fugacity=high_fugacities[group],
                inert_fugacity=0.0,
                inert_to_bonding_rate=0.0,
                area=100*100,
                stopping_size=100,
                match_mean=np.average(times[group]),
            ),
            "--",
            color=color,
        )

        ax1s[i].legend()
        ax1s[i].set(xlim=[0, np.average(times[group]) * upper_time_limit])
        ax1s[i].set(xlabel="$t$", ylabel="Density")

    plt.show()


def cnt_predicted_pdf(
    ts,
    beta,
    bond_energy,
    bonding_fugacity,
    inert_fugacity,
    inert_to_bonding_rate,
    stopping_size,
    area,
    drift_diffusion_approx_size=None,
    match_mean=None,
):
    bonding_potential = np.log(bonding_fugacity / (1 + inert_fugacity))
    monomer_concentration = bonding_fugacity / (1 + inert_fugacity + bonding_fugacity)
    magnetic_field, bond_energy = (
        -bond_energy + bonding_potential / 2,
        -bond_energy / 4,
    )

    condensed_concentration = np.power(
        1 - 1 / (np.sinh(2 * beta * bond_energy) ** 4),
        1/8
    )
    particle_circumference = 2 * np.sqrt(np.pi / condensed_concentration)
    single_particle_free_energy = 8 * bond_energy

    beta_surface_tension = 2 * beta * bond_energy + np.log(
        np.tanh(beta * bond_energy)
    ) + np.sqrt(2) * np.log(
        np.sinh(beta * bond_energy)
    )
    tau = 5/4

    dimensionless_tension = particle_circumference * beta_surface_tension / 4

    critical_size = (
        dimensionless_tension / beta / magnetic_field
        + np.sqrt(
            (dimensionless_tension / beta / magnetic_field) ** 2
            + tau / beta / magnetic_field
        )
    ) ** 2

    print(f"\ti_crit\t\t= {critical_size:.1f}")

    def forward_rate(size, field=0):
        return (
            inert_to_bonding_rate * inert_fugacity + 1
        ) * np.exp(field / beta) * np.exp(
            - 2 * beta * bond_energy
        ) * 2 * np.sqrt(np.pi * size / condensed_concentration)

    cnt_rate = monomer_concentration * area * forward_rate(
        critical_size
    ) * np.sqrt((
        dimensionless_tension * np.power(critical_size, -3/2)
        + tau / (critical_size**2)
    ) / 2 * np.pi) / np.exp(
        2 * dimensionless_tension ** 2 / beta / magnetic_field
        + 2 * dimensionless_tension * np.sqrt(
            dimensionless_tension ** 2 / beta ** 2 / magnetic_field ** 2
            + tau / beta / magnetic_field
        )
        - 4 * dimensionless_tension
        - tau
        + beta * 8 * bond_energy
        + tau / 2 * np.log(
            dimensionless_tension / beta / magnetic_field
            + np.sqrt(
                dimensionless_tension ** 2 / beta ** 2 / magnetic_field ** 2
                + tau / beta / magnetic_field
            )
        )
    )

    print(f"\t1/cnt_rate\t= {1/cnt_rate:.1f}")
    

    def beta_driving_force(size):
        return (
            1 / 2 / np.sqrt(size) * dimensionless_tension
            + tau / size - magnetic_field
        )

    def drift(size):
        return forward_rate(size, magnetic_field) * (
            - beta_driving_force(size)
            + 1 / 2 / size
        )

    def diffusion(size):
        return forward_rate(size, magnetic_field)

    delta_size = stopping_size - critical_size

    if match_mean is not None:
        search_range = [critical_size, stopping_size]
        print(f"searching for {match_mean}")
        while True:
            pred_mean = (
                1 / drift(np.average(search_range)) * delta_size
                + 1/cnt_rate
            )
            print(f"search_range: {search_range}, pred_mean: {pred_mean:.1f}")
            error = (pred_mean - match_mean) / match_mean
            if abs(error) < 0.01:
                break
            elif error > 0:
                search_range[0] = np.average(search_range)
            else:
                search_range[1] = np.average(search_range)
            if abs(search_range[0] - search_range[1]) / search_range[0] < 1e-6:
                break
        drift_diffusion_approx_size = np.average(search_range)
        print(f"critical size = {critical_size:.1f}, stopping size = {stopping_size:.0f}, drift_diffusion_approx_size = {drift_diffusion_approx_size:.1f}")
    elif drift_diffusion_approx_size is None:
        drift_diffusion_approx_size = np.sqrt(
            critical_size * stopping_size
        )

    print(f"\t1/drift_min\t= {1/drift(stopping_size) * delta_size:.3f}")
    print(f"\t1/drift_max\t= {1/drift(critical_size) * delta_size:.3f}")

    # NOTE(Myles): This function has numerical problems
    def p_exact(ts, size=drift_diffusion_approx_size):
        if (drift(size) / delta_size) ** 2 - 4 * cnt_rate * diffusion(size) / delta_size**2 < 0:
            print("Big Problem! Effective Drift is Negative")

        return cnt_rate * np.exp(
            (
                drift(size) / delta_size
                - np.sqrt((drift(size) / delta_size) ** 2
                          - 4 * cnt_rate * diffusion(size) / delta_size**2)
            ) / diffusion(size) * delta_size**2
            - cnt_rate * ts
        ) * scipy.stats.invgauss.cdf(
            ts,
            mu=2*diffusion(size)/delta_size/np.sqrt((drift(size) / delta_size) ** 2 - 4 * cnt_rate * diffusion(size) / delta_size**2),
            loc=0,
            scale=1/2/diffusion(size) * delta_size**2,
        )

    def p(ts, size=drift_diffusion_approx_size):
        return scipy.stats.norm.pdf(
            ts,
            loc=1/cnt_rate + 1/drift(size) * delta_size,
            scale=np.sqrt(1/cnt_rate**2 + 2 * diffusion(size)/drift(size)**3 * delta_size),
        )

    return p(ts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("effect_of_activity")
    parser.add_argument(
        "-r", "--run",
        action="store_true",
        help="Run simulations",
    )
    parser.add_argument(
        "-a", "--analyze",
        required=False,
        nargs="?",
        const=DEFAULT_DIR,
        help="Run analysis on previously run simulations stored at PATH",
        metavar="PATH",
    )
    args = parser.parse_args()

    if args.run:
        run()

    if args.analyze is not None:
        analyze(args.analyze)

    # z_B_coex = 0.002724609375
    # low_fugacities = [z_B_coex / 9, z_B_coex / 3]
    # high_fugacities = [z_B_coex * 3, z_B_coex * 9]
    
    # for high_fugacity in high_fugacities:
    #     print(f"z_b = {high_fugacity:.4f}")
        # cnt_predicted_pdf(
        #     beta=1.0,
        #     bond_energy=-3.0,
        #     bonding_potential=np.log(high_fugacity),
        #     inert_potential=np.log(0.1),
        #     inert_to_bonding_rate=0.1,
        #     area=100*100,
        #     stopping_size=9500,
        # )
    # plt.show()
