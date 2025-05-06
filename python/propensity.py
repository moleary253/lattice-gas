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


def analyze(
        tar_archive=TARFILE,
        directory=DIRECTORY,
        independent_variable="bonding_fugacity",
):
    load.unpack_tar_archive(tar_archive, directory)
    try:

        simulations = [
            os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS, path)
            for path in os.listdir(os.path.join(directory, PATH_TO_COEXISTENCE_SIMULATIONS))
        ]

        times = {}
        for simulation in tqdm(simulations):
            chain_json = load.chain(simulation)
            sizes = load.sizes(simulation)

            time = sizes[-1,0]

            if chain_json[independent_variable] not in times:
                times[chain_json[independent_variable]] = []

            times[chain_json[independent_variable]].append(time)

        plot(times, independent_variable)

    finally:
        shutil.rmtree(directory)


def plot(times, independent_variable):
    label = {
        "driving_chemical_potential": "$\\Delta \\mu$",
        "bonding_fugacity": "$z_B$",
    }[independent_variable]

    fig1, ax1s = plt.subplots(5, 1)

    keys = np.array(sorted(list(times.keys())))
    sorts = [np.argsort(times[key]) for key in keys]
    times = np.array([np.array(times[key])[sorts[i]] for i, key in enumerate(keys)])
    times = times/np.average(times, axis=1).reshape((-1, 1))
    pdfs = [gaussian_kde(time) for time in times]
    probabilities = (1 + np.arange(len(times[0]))) / len(times[0])

    print(np.std(times, axis=1)/np.average(times, axis=1))

    upper_time_limit = 2

    bins = np.linspace(0, upper_time_limit, 15)
    sample_times = np.linspace(np.min(bins), np.max(bins), 100)
    for i in range(len(times[:,0])):
        color = (0.2 * i, 0., 0.8 - 0.2 * i, 0.5)
        n, x, _ = ax1s[i].hist(times[i], bins=bins, 
                           histtype=u'bar',
                           density=True,
                           label=f"{keys[i]:.2f}",
                           color=color)
        ax1s[i].plot(sample_times, pdfs[i](sample_times), label=f"{keys[i]:.2f}", color=color)

        ax1s[i].legend()
        ax1s[i].set(xlim=[0, upper_time_limit])
        ax1s[i].set(xlabel="$t$", ylabel="Density")

    fig3, ax3 = plt.subplots()

    ks = np.array([[pdf(t)/(1.01 - pdf.integrate_box(0, t)) for t in sample_times] for pdf in pdfs])
    for i in range(ks.shape[0]):
        ax3.plot(sample_times, ks[i], color=f"C{i+1}", label=f"{keys[i]:.2f}")

    ax3.legend()
    ax3.set(xlim=[0, upper_time_limit])
    ax3.set(xlabel="$t$", ylabel="$k$")

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("propensity")
    parser.add_argument("-f", "--filename", default=TARFILE)
    parser.add_argument("--independent-variable", default="bonding_fugacity")
    args = parser.parse_args()

    analyze(args.filename, independent_variable=args.independent_variable)
