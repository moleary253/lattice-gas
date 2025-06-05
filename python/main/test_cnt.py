import lattice_gas as lg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import load


DEFAULT_DIR="data/test_cnt"


def run_simulation(output_file, random_seed, magnetic_field, bond_energy, initial_state, ending_criteria):
    boundary = lg.boundary_condition.Periodic()

    chain = lg.markov_chain.IsingChain(
        beta=1.0,
        bond_energy=bond_energy,
        magnetic_field=magnetic_field,
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
    bond_energy = -3.0
    magnetic_fields = [5, 3]
    
    if os.path.exists(DEFAULT_DIR):
        shutil.rmtree(DEFAULT_DIR)
    os.mkdir(DEFAULT_DIR)
    for i in tqdm(range(num_trials)):
        for magnetic_field in magnetic_fields:
            random_seed = 17 + i * 31 + int(magnetic_field * 1e5) % 11
            output_file = DEFAULT_DIR + f"/{magnetic_field:.5f}_{i+1}.tar.gz"
            initial_state = np.zeros((100, 100), dtype=np.dtype("u4"))

            ending_criterion = lg.ending_criterion.LargestDropletSize(
                100,
                [load.BONDING],
            )
            failsafe_ending_criterion = lg.ending_criterion.ReactionCount(
                1_000_000
            )
            run_simulation(
                output_file,
                random_seed,
                magnetic_field,
                bond_energy,
                initial_state,
                [ending_criterion, ],
            )


def analyze(path):
    from scipy.stats import gaussian_kde

    boundary = lg.boundary_condition.Periodic()

    times = {}
    threshold = 10
    magnetic_fields = {}
    for sim_name in tqdm(os.listdir(path)):
        sim_path = os.path.join(path, sim_name)
        directory = load.unpack_natural_input(sim_path)

        magnetic_field = float(sim_name.split("_")[0])
        group = f"$h={magnetic_field:.5f}$"
        if group not in times:
            times[group] = []
            magnetic_fields[group] = magnetic_field

            
        state = initial_state.copy().astype("u4")
        reactions = load.reactions()
        droplets = lg.analysis.Droplets(
            state,
            boundary,
            [load.BONDING],
        )
        t = 0
        for reaction in reactions:
            load.apply_reaction(state, reaction)
            droplets.update(
                state,
                boundary,
                [load.BONDING],
                reaction,
            )
            t += reaction["dt"]
            if np.max([len(drop) for drop in droplets.droplets()]) == threshold:
                break
        time = t

        time = load.final_time()
        times[group].append(time)
        load.clean_up_temp_files()
        
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
                magnetic_field=magnetic_fields[group],
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
    magnetic_field,
    stopping_size,
    area,
    drift_diffusion_approx_size=None,
    match_mean=None,
):
    assert bond_energy < 0

    monomer_concentration = 1 / (1 + np.exp(-magnetic_field * beta - 4 * beta * bond_energy))
    print(f"\tmonomer conc = {monomer_concentration}")

    condensed_concentration = np.power(
        1 - 1 / (np.sinh(2 * beta * bond_energy) ** 4),
        1/8
    )
    print(f"\tcondensed conc = {condensed_concentration}")
    particle_circumference = 2 * np.sqrt(np.pi / condensed_concentration)
    print(f"\tparticle circ= {particle_circumference}")

    beta_surface_tension = - 2 * beta * bond_energy + np.log(
        np.tanh(-beta * bond_energy)
    ) + np.sqrt(2) * np.log(
        np.sinh(-beta * bond_energy)
    )
    print(f"\tbeta_surface_tension = {beta_surface_tension}")
    tau = 5/4

    dimensionless_tension = particle_circumference * beta_surface_tension / 4
    print(f"\tdimensionless_surface_tension = {dimensionless_tension}")

    critical_size = (
        dimensionless_tension / beta / magnetic_field
        + np.sqrt(
            (dimensionless_tension / beta / magnetic_field) ** 2
            + tau / beta / magnetic_field
        )
    ) ** 2

    print(f"\ti_crit\t\t= {critical_size:.1f}")

    def forward_rate(size, field=0):
        return np.exp(field * beta) * np.exp(
            2 * beta * bond_energy
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

    if match_mean is not None and cnt_rate != 0:
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
