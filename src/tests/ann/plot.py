#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.animation import FuncAnimation
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 1):
        print(f"USAGE: {sys.argv[0]} file.csv", file=sys.stderr)
    path = Path(sys.argv[1])
    with open(path, "r") as file:
        train_x = [float(d.strip()) for d in file.readline().split(',')]
        train_y = [float(d.strip()) for d in file.readline().split(',')]
        predictions = []
        prediction_in = [f.strip() for f in file.readline().split(',') if len(f) > 0]
        while prediction_in:
            predictions.append([float(d) for d in prediction_in])
            prediction_in = [f.strip() for f in file.readline().split(',') if len(f) > 0]

    fig, ax = plt.subplots()
    ln, = ax.plot([], [], color="red", lw=3, label='Predicted values')
    epochs = list(range(0, len(predictions)))

    def init():
        ax.set_xlim(train_x[0], train_x[-1])
        ax.set_ylim(min(*predictions[0]) - 1, max(*predictions[0]) + 1)
        return ln,

    def update(epoch):
        ln.set_data(train_x, predictions[epoch])
        ln.set_label(f"Predicted values after {epoch} epochs")
        return ln,

    ax.set_title(f"Results")
    ax.plot(train_x, train_y, "blue", lw=4, label='Expected values')
    ax.legend()

    ani = FuncAnimation(fig, update, frames=epochs, interval=33,
                        init_func=init, blit=True)

    #ani.save('predictions.mp4', fps=30, dpi=200)
    plt.show()
