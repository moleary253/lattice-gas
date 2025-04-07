import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.stats import gaussian_kde
from scipy import ndimage

import sys, os
import shutil
import tarfile
import json

from tqdm import tqdm

import load


TARFILE = os.path.join("data", "propensity.tar.gz")
DIRECTORY = os.path.join("data", "propensity")
PATH_TO_COEXISTENCE_SIMULATIONS = "propensity"
OUTCOME_FILE = "outcome.txt"


def analyze(tar_archive=TARFILE, directory=DIRECTORY):
    load.unpack_tar_archive(tar_archive, directory)
    try:

        simulations = [
            os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS, path)
            for path in os.listdir(os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS))
        ]

        times = {}
        droplet_counts = {}
        independent_variable = "kinase_chemical_potential"
        for simulation in tqdm(simulations):
            if not os.path.exists(os.path.join(simulation, OUTCOME_FILE)):
                continue

            chain_json, initial_conditions, reactions = load.directory(simulation)

            with open(os.path.join(simulation, OUTCOME_FILE)) as f:
                data = f.read()

            time = float(data[1:-1])

            if chain_json[independent_variable] not in times:
                times[chain_json[independent_variable]] = []
                droplet_counts[chain_json[independent_variable]] = []

            droplet_counts[chain_json[independent_variable]].append(0)
            times[chain_json[independent_variable]].append(time)

            state = initial_conditions.copy()
            for reaction in reactions:
                load.apply_reaction(state, reaction)

            state[state == load.INERT] = load.EMPTY

            labeled_array, num_features = ndimage.label(state, np.ones((3,3)))
            for i in range(num_features):
                only_one_particle = np.sum(labeled_array == i + 1) == 1
                if not only_one_particle:
                    droplet_counts[chain_json[independent_variable]][-1] += 1

        plot(times, droplet_counts)

    finally:
        shutil.rmtree(directory)


def plot(times, droplet_counts):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    keys = np.array(sorted(list(times.keys())))
    sorts = [np.argsort(times[key]) for key in keys]
    times = np.array([np.array(times[key])[sorts[i]] for i, key in enumerate(keys)])
    droplet_counts = np.array([np.array(droplet_counts[key])[sorts[i]] for i, key in enumerate(keys)])
    pdfs = [gaussian_kde(time) for time in times]
    probabilities = (1 + np.arange(len(times[0]))) / len(times[0])

    upper_time_limit = 5000

    bins = np.linspace(0, upper_time_limit, 40)
    sample_times = np.linspace(np.min(bins), np.max(bins), 100)
    for i in range(len(times[:,0])):
        n, x, _ = ax1.hist(times[i], bins=bins, 
                           histtype=u'step', density=True, label=f"{keys[i]:.2f}",
                           color=f"C{i+1}")
        ax2.plot(sample_times, pdfs[i](sample_times), label=f"{keys[i]:.2f}", color=f"C{i+1}")

    ax1.legend()
    ax1.set(xlim=[0, upper_time_limit])
    ax1.set(xlabel="$t$", ylabel="Density")

    ax2.legend()
    ax2.set(xlim=[0, upper_time_limit])
    ax2.set(xlabel="$t$", ylabel="$\\hat p(t)$")

    fig3, ax3 = plt.subplots()

    ks = np.array([[pdf(t)/(1.01 - pdf.integrate_box(0, t)) for t in sample_times] for pdf in pdfs])
    for i in range(ks.shape[0]):
        ax3.plot(sample_times, ks[i], color=f"C{i+1}", label=f"{keys[i]:.2f}")

    ax3.legend()
    ax3.set(xlim=[0, upper_time_limit])
    ax3.set(xlabel="$t$", ylabel="$k$")

    fig4, ax4 = plt.subplots()
    ax4.errorbar(keys, np.average(droplet_counts, axis=1), np.std(droplet_counts, axis=1)/np.sqrt(droplet_counts.shape[1]), capsize=4.0, fmt="o")
    ax4.set(ylim=[0, 2])
    ax4.set(xlabel="$\\Delta \\mu$", ylabel="Average Num Droplets")

    fig5, ax5 = plt.subplots()
    bins = np.arange(10)
    for i in range(droplet_counts.shape[0]):
        ax5.hist(droplet_counts[i], bins=bins, 
                 histtype=u'step', density=True, label=f"{keys[i]:.2f}",
                 color=f"C{i+1}")

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("propensity")
    parser.add_argument("-f", "--filename", default=TARFILE)
    args = parser.parse_args()

    analyze(args.filename)
