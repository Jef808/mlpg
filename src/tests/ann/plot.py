#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

path = Path("../../../data")


if __name__ == '__main__':
    x = np.arange(-2*math.pi, 2*math.pi, 0.1)

    with open(path / Path(f"training0.csv"), "r") as file:
        train_x_in = [f.strip() for f in file.readline().split(',')]
        train_x = [float(d) for d in train_x_in if len(d) > 0]

        train_y_in = [f.strip() for f in file.readline().split(',')]
        train_y = [float(d) for d in train_y_in if len(d) > 0]

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12,6))
        ax0.set_title(f"training data")
        ax0.errorbar(train_x, train_y)

        for i in range(30):
            with open(path / Path(f"sample{i}.csv"), "r") as file:
                sample_x_in = [f.strip() for f in file.readline().split(',')]
                sample_x = [float(d) for d in train_x_in if len(d) > 0]

                sample_y_in = [f.strip() for f in file.readline().split(',')]
                sample_y = [float(d) for d in train_y_in if len(d) > 0]

                ax1.set_title(f"sample # {i}")
                ax1.errorbar(sample_x, sample_y, errorevery=6)
                plt.savefig(f"data/plot{i}.png")
