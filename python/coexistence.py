import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.stats import binomtest

import sys, os
import shutil
import tarfile
import json

import load


TARFILE = os.path.join("data", "coexistence.tar.gz")
DIRECTORY = os.path.join("data", "coexistence")
PATH_TO_COEXISTENCE_SIMULATIONS = "coexistence"
OUTCOME_FILE = "outcome.txt"


def analyze(tar_archive=TARFILE, directory=DIRECTORY):
    load.unpack_tar_archive(tar_archive, directory)

    simulations = [
        os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS, path)
        for path in os.listdir(os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS))
    ]

    n_gas = {}
    counts = {}
    independent_variable = "driving_chemical_potential"
    for simulation in simulations:
        if not os.path.exists(os.path.join(simulation, OUTCOME_FILE)):
            continue

        with open(os.path.join(simulation, load.CHAIN_FILE)) as f:
            chain_json = json.load(f)

        with open(os.path.join(simulation, OUTCOME_FILE)) as f:
            data = f.read()

        print(chain_json)

        if chain_json[independent_variable] not in counts:
            counts[chain_json[independent_variable]] = 0
            n_gas[chain_json[independent_variable]] = 0

        counts[chain_json[independent_variable]] += 1
        if "gas" in data:
            n_gas[chain_json[independent_variable]] += 1

    plot(counts, n_gas)

    shutil.rmtree(DIRECTORY)


def plot(counts, n_gas):
    fig, ax = plt.subplots()
    
    keys = np.array(sorted(list(counts.keys())))
    counts = np.array([counts[key] for key in keys])
    n_gas = np.array([n_gas[key] for key in keys])
    fractions_gas = n_gas / counts
    chemical_potentials = keys

    # Uses exact method
    intervals_fractions_gas = np.abs(np.array([
        [result.proportion_ci(0.68).low, result.proportion_ci(0.68).high]
        for result in [binomtest(n_gas[i], counts[i], p=fractions_gas[i], alternative="two-sided") for i in range(len(keys))]
    ]).T - fractions_gas)

    # Wald method to get sigmas for the fit
    sigmas_fractions_gas = np.sqrt(fractions_gas * (1 - fractions_gas) / counts)

    for i, chemical_potential in enumerate(chemical_potentials):
        print(f"{chemical_potential}: {n_gas[i]}/{counts[i]} +{intervals_fractions_gas[1,i]:.3f}-{intervals_fractions_gas[0,i]:.3f}")

    ax.errorbar(chemical_potentials, fractions_gas, yerr=intervals_fractions_gas, fmt=".", capsize=2.0)

    def sigmoid(x, midpoint, decay_rate):
        return 1 / (1 + np.exp(decay_rate * (x - midpoint)))

    optimal_params, covariance = scipy.optimize.curve_fit(
        sigmoid,
        chemical_potentials,
        fractions_gas,
        p0=[np.mean(chemical_potentials), 10],
        sigma=sigmas_fractions_gas,
        absolute_sigma=True
    )

    print(f"midpoint: {optimal_params[0]:.3f} +- {np.sqrt(covariance[0,0]):.4f}, decay_rate: {optimal_params[1]:.2f} +- {np.sqrt(covariance[1,1]):.4f}")

    fit_xs = np.linspace(np.min(chemical_potentials), np.max(chemical_potentials), 100)
    ax.plot(fit_xs, sigmoid(fit_xs, *optimal_params), "--")

    ax.set(xlabel="$\\mu_B$", ylabel="p(gas)")
    ax.set(ylim=[0,1])

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("plot")
    parser.add_argument("-f", "--filename", default=TARFILE)
    args = parser.parse_args()

    analyze(args.filename)
