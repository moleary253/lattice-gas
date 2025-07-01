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

    analyzers = [
        lg.analysis.Droplets(
            initial_state,
            boundary,
            [load.BONDING],
        ),
        lg.analysis.LargestDropletSizeAnalyzer(),
    ]


    lg.simulate.simulate(
        initial_state,
        boundary,
        chain,
        analyzers,
        ending_criteria,
        random_seed,
        output_file,
    )


def run():
    import shutil

    num_trials = 100
    bond_energy = -1.5
    magnetic_fields = [1.10, 1.15, 1.20, 1.25]
    
    if os.path.exists(DEFAULT_DIR):
        shutil.rmtree(DEFAULT_DIR)
    os.mkdir(DEFAULT_DIR)
    for i in tqdm(range(num_trials)):
        for magnetic_field in magnetic_fields:
            random_seed = 17 + i * 31 + int(magnetic_field * 1e5) % 11
            output_file = DEFAULT_DIR + f"/{magnetic_field:.5f}_{i+1}"
            initial_state = np.zeros((100, 100), dtype=np.dtype("u4"))

            ending_criterion = lg.ending_criterion.LargestDropletSize(
                3000,
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


def forward_rate(size, bond_energy, field=0):
    return np.exp(
        field + 5 * bond_energy
    ) * (2 * np.sqrt(np.pi) * (np.sqrt(size) - 1))


def analyze(path):
    from scipy.stats import gaussian_kde

    random_sim = os.listdir(path)[0]
    random_sim = os.path.join(path, random_sim)
    boundary = lg.boundary_condition.Periodic()

    bond_energy = -1.5

    times = {}
    magnetic_fields = {}
    bot_absorb_size = 1
    try:
        ending_criteria = lg.load.ending_criteria(random_sim)
        top_absorb_size = ending_criteria[0].threshold
    except:
        top_absorb_size = 100
    time_seen = {}
    time_succeeded = {}
    time_seen_rates = {}
    num_fwd = {}
    num_bkwd = {}
    for sim_name in tqdm(os.listdir(path)):
        sim_path = os.path.join(path, sim_name)

        magnetic_field = float(sim_name.split("_")[0])
        group = f"$h={magnetic_field:.5f}$"
        if group not in times:
            times[group] = []
            magnetic_fields[group] = magnetic_field
            time_seen[group] = np.zeros(top_absorb_size - bot_absorb_size - 1)
            time_succeeded[group] = np.zeros(time_seen[group].shape)
            time_seen_rates[group] = np.zeros(top_absorb_size + 1)
            num_fwd[group] = np.zeros(top_absorb_size)
            num_bkwd[group] = np.zeros(top_absorb_size)
            
        final_time = lg.load.final_time(sim_path)

        analyzers = lg.load.analyzers(sim_path)
        largest_size = analyzers[1]
        t_succeeded, t_seen = lg.analysis.commitance(
            largest_size.sizes,
            largest_size.delta_times,
            bot_absorb_size,
            top_absorb_size,
        )
        time_seen[group] += t_seen
        time_succeeded[group] += t_succeeded

        t_seen_rates, n_fwd, n_bkwd = lg.analysis.cnt_rates(
            largest_size.sizes,
            largest_size.delta_times,
        )
        time_seen_rates[group] += t_seen_rates[:time_seen_rates[group].size]
        num_fwd[group] += n_fwd[:num_fwd[group].size]
        num_bkwd[group] += n_bkwd[:num_bkwd[group].size]

        times[group].append(final_time)

    fig1, ax1s = plt.subplots(len(times), 1)

    upper_time_limit = 2
    bins = np.linspace(0, upper_time_limit, 90)
    sample_times = np.linspace(np.min(bins), np.max(bins), 300)

    pdfs = [gaussian_kde(times[group]) for group in times]
    cnt_pdfs = {}
    crit_sizes = {}
    for group in times:
        pdf, crit_size = cnt_predicted_pdf(
            np.average(times[group]) * sample_times,
            beta=1.0,
            bond_energy=bond_energy,
            magnetic_field=magnetic_fields[group],
            area=100*100,
            stopping_size=100,
            match_mean=np.average(times[group]),
        )
        cnt_pdfs[group] = pdf
        crit_sizes[group] = crit_size

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
            cnt_pdfs[group],
            "--",
            color=color,
        )

        ax1s[i].legend()
        ax1s[i].set(xlim=[0, np.average(times[group]) * upper_time_limit])
        ax1s[i].set(xlabel="$t$", ylabel="Density")

    fig2, ax2 = plt.subplots()
    for i, group in enumerate(time_succeeded):
        ax2.plot(
            np.arange(bot_absorb_size + 1, top_absorb_size),
            time_succeeded[group] / time_seen[group],
            f"C{i}-",
        )
        ax2.plot(
            [crit_sizes[group], crit_sizes[group]],
            [0, 1],
            f"C{i}--",
        )
    ax2.plot(
        [bot_absorb_size + 1, top_absorb_size],
        [0.5, 0.5],
        "r--",
    )

    fig3, ax3s = plt.subplots(len(times), 1)
    for i, group in enumerate(times):
        color = (
            0.8 * i / (len(times) - 1),
            0.,
            0.8 * (1 - i / (len(times) - 1)),
            0.5
        )
        xs = np.arange(0, len(num_fwd[group]))
        particle_circumference = 2 * np.sqrt(np.pi)
        ax3s[i].plot(
            xs[1:],
            num_fwd[group][1:] / time_seen_rates[group][1:-1],
            ".",
            label=group + " $k_f$",
            color=color
        )

        ax3s[i].plot(
            xs,
            forward_rate(xs, bond_energy, field=magnetic_fields[group]),
            "--",
            label=group + " $k_f$",
            color=color
        )

        ax3s[i].legend()
        ax3s[i].set(xlabel="$i$", ylabel="Rate (1/time)")

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

    print(f"\tbond_energy = {bond_energy}")
    print(f"\tmagnetic_field = {magnetic_field}")

    condensed_concentration = np.power(
        1 - 1 / (np.sinh(2 * beta * bond_energy) ** 4),
        1/8
    )
    print(f"\tcondensed conc = {condensed_concentration}")
    particle_circumference = 2 * np.sqrt(np.pi / condensed_concentration)
    print(f"\tparticle circ = {particle_circumference}")

    beta_surface_tension = (- 2 * beta * bond_energy + np.log(
        np.tanh(-beta * bond_energy)
    ) + np.sqrt(2) * np.log(
        np.sinh(-2 * beta * bond_energy)
    )) / (2 * condensed_concentration)
    print(f"\tbeta_surface_tension = {beta_surface_tension}")
    tau = 5/4

    dimensionless_tension = particle_circumference * beta_surface_tension / 4
    print(f"\tdimensionless_surface_tension = {dimensionless_tension}")

    critical_size = (
        dimensionless_tension / beta / 2 / magnetic_field
        + np.sqrt(
            (dimensionless_tension / beta / 2 / magnetic_field) ** 2
            + tau / beta / 2 / magnetic_field
        )
    ) ** 2

    print(f"\ti_crit\t\t= {critical_size:.1f}")

    monomer_concentration = 1 / (1 + np.exp(-2 * magnetic_field * beta - 8 * beta * bond_energy))
    print(f"\tmonomer conc = {monomer_concentration}")

    def cnt_rate_exponent(size):
        return np.exp(
            (1 - np.sqrt(size)) * particle_circumference * beta_surface_tension
            + 8 * beta * bond_energy
            - tau * np.log(size)
            + 2 * size * beta * magnetic_field
        )
    print(f"\t-beta g(i*)\t= {np.log(cnt_rate_exponent(critical_size)):.1f}")

    prefactor = (1
        # * monomer_concentration
        * area
        * forward_rate(
            critical_size, bond_energy, field=magnetic_field,
        )
        * np.sqrt((
            dimensionless_tension * np.power(critical_size, -3/2)
            + tau / (critical_size**2)
        ) / 2 / np.pi)
    )
    print(f"\tcnt_prefactor\t= {prefactor:.5f}")
    print(f"\tln(cnt_prefactor)\t= {np.log(prefactor):.5f}")
    print(f"\tln(cnt_prefactor/monomer_concentration)\t= {np.log(prefactor/monomer_concentration):.5f}")

    cnt_rate = prefactor * cnt_rate_exponent(critical_size)
    print(f"\t1/cnt_rate\t= {1/cnt_rate:.1f}")
    

    def beta_driving_force(size):
        return (
            1 / 2 / np.sqrt(size) * dimensionless_tension
            + tau / size - 2 * magnetic_field
        )

    def drift(size):
        return forward_rate(size, bond_energy, magnetic_field) * (
            - beta_driving_force(size)
            + 1 / 2 / size
        )

    def diffusion(size):
        return forward_rate(size, bond_energy, magnetic_field)

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

    return p(ts), critical_size


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
