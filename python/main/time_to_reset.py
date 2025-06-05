import lattice_gas as lg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import load


DEFAULT_DIR="data/time_to_reset"


def run_simulation(output_file, random_seed, bonding_fugacity, initial_state, ending_criteria):
    boundary = lg.boundary_condition.Periodic()

    chain = lg.markov_chain.HomogenousChain(
        beta=1.0,
        bond_energy=-3.0,
        driving_chemical_potential=0.0,
        inert_fugacity=0.1,
        bonding_fugacity=bonding_fugacity,
        inert_to_bonding_rate=0.1,
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


def run(continue_from):
    import shutil

    if os.path.exists(DEFAULT_DIR):
        shutil.rmtree(DEFAULT_DIR)
    os.mkdir(DEFAULT_DIR)

    for sim_name in tqdm(os.listdir(continue_from)):
        sim_path = os.path.join(continue_from, sim_name)

        low_fugacity = float(sim_name.split("_")[0])
        high_fugacity = float(sim_name.split("_")[2])
        sim_id = int(sim_name.split("_")[3].split(".")[0])

        directory = load.unpack_natural_input(sim_path)
        old_final_state = load.final_state()
        load.clean_up_temp_files()

        random_seed = np.random.randint(0, 2 ** 63)
        output_file = os.path.join(
            DEFAULT_DIR,
            f"{low_fugacity:.5f}_to_{high_fugacity:.4}_to_{low_fugacity:.5f}_{sim_id}.tar.gz"
        )

        gas_density = 0.01
        gas_ending_criterion = lg.ending_criterion.ParticleCount(
            int(old_final_state.size * gas_density),
            [load.BONDING],
            "below"
        )
        run_simulation(
            output_file,
            random_seed,
            low_fugacity,
            old_final_state,
            [gas_ending_criterion]
        )


def analyze(path):
    from scipy.stats import gaussian_kde
    def is_bonding(x):
        return x == load.BONDING

    boundary = lg.boundary_condition.Periodic()

    times = {}
    for sim_name in tqdm(os.listdir(path)):
        sim_path = os.path.join(path, sim_name)
        directory = load.unpack_natural_input(sim_path)

        low_fugacity = float(sim_name.split("_")[0])
        high_fugacity = float(sim_name.split("_")[2])
        delta_mu = np.log(high_fugacity / low_fugacity)
        group = f"$z_B={high_fugacity:.5f}$ to $z_B={low_fugacity:.4f}$"
        if group not in times:
            times[group] = []

        initial_state = load.initial_conditions()

        time = load.final_time()
        times[group].append(time)
        load.clean_up_temp_files()
        

    fig1, ax1 = plt.subplots()

    upper_time_limit = np.max([np.max(times[group]) for group in times]) * 1.1
    bins = np.arange(0, upper_time_limit, 10)
    sample_times = np.linspace(np.min(bins), np.max(bins), len(bins) * 10)

    pdfs = [gaussian_kde(times[group]) for group in times]

    for i, group in enumerate(times):
        color = (
            0.8 * i / (len(times) - 1),
            0.,
            0.8 * (1 - i / (len(times) - 1)),
            0.5
        )
        n, x, _ = ax1.hist(
            times[group], bins=bins, 
            histtype=u'bar',
            density=True,
            label=group,
            color=color
        )
        ax1.plot(
            sample_times,
            pdfs[i](sample_times),
            color=color
        )

    ax1.legend()
    ax1.set()
    ax1.set(xlabel="$t$", ylabel="Density")

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run",
        required=False,
        help="Run simulations, continuing from those at PATH",
        metavar="PATH",
    )
    parser.add_argument(
        "-a", "--analyze",
        nargs="?",
        const=DEFAULT_DIR,
        help="Run analysis on previously run simulations stored at PATH",
        metavar="PATH",
        required=False,
    )
    args = parser.parse_args()

    if args.run is not None:
        run(args.run)

    if args.analyze is not None:
        analyze(args.analyze)
