import lattice_gas as lg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import load


DEFAULT_DIR="data/step_j"


def run():
    pass

def analyze(directory=DEFAULT_DIR):
    low_rate = 2
    high_rate = 10
    switch_time = 0.2
    
    us = np.random.random_sample(3000)
    ts = np.empty(us.size)
    low_mask = us > np.exp(- low_rate * switch_time)
    high_mask = us < np.exp(- low_rate * switch_time)
    ts[low_mask] = -1/low_rate * np.log(us[low_mask])
    ts[high_mask] = -1/high_rate * (
        np.log(us[high_mask])
        + low_rate * switch_time
        - high_rate * switch_time
    )


    ks = np.exp(np.linspace(-5, 5, 101))
    tries = 10
    cgfs = np.empty(ks.size)
    cgf_stds = np.empty(ks.size)
    for i, k in enumerate(ks):
        sim_ts = np.array([
            lg.simulate.calculate_cgf(ts, k, seed=1024 + 13 * i + int(17 * k) + 11 * j)
            for j
            in range(tries)
        ])
        unbiased_estimators = np.exp(-k * np.average(sim_ts, axis=1))
        unbiased_std = np.std(unbiased_estimators, ddof=1)
        unbiased_mean = np.average(unbiased_estimators)
        cgfs[i] = np.log(unbiased_mean)
        cgf_stds[i] = unbiased_std / np.sqrt(tries) / unbiased_mean
        
    true_cgfs = (
        -np.log(1+ks/low_rate)
        + np.log(
            1 +
            (-1 + (1+ks/low_rate) / (1+ks/high_rate)) 
            * np.exp((-low_rate - ks) * switch_time))
    )

    fig, ax = plt.subplots()
    # plt.plot(ks, -np.average(ts) * ks, "--", label="First Order Talyor Approx")
    plt.errorbar(ks, cgfs, yerr=cgf_stds, fmt=".", capsize=1, label="Simulated")
    plt.plot(ks, true_cgfs, label="Analytical")
    plt.legend()

    fig, ax = plt.subplots()
    for i, offset in enumerate(range(1, 31, 5)):
        energy_gap = np.average(np.log(ks[:-offset] / ks[offset:]))
        ax.errorbar(
            cgfs[offset:],
            cgfs[:-offset],
            xerr=cgf_stds[offset:],
            yerr=cgf_stds[offset:],
            fmt=f"C{i}.",
            label=f"$\\Delta={energy_gap:.2f}$"
        )
        ax.plot(true_cgfs[offset:], true_cgfs[:-offset], f"C{i}-", label="_")
    ax.set(xlabel="Log Odds Wrong", ylabel="Log Odds Correct")
    ax.legend()

    plt.show()


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
