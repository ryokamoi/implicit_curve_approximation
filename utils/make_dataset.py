import os
import sys
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def make_dataset(outputfilename: str) -> None:
    xlim = [-1.5, 1.5]
    ylim = [1.5, -1.5]

    fig=plt.figure()

    if len(sys.argv) >= 3:
        figname = sys.argv[2]
        background = plt.imread(figname)
        plt.imshow(background, extent=[xlim[0], xlim[1], ylim[1], ylim[0]])

    plt.xlim(left=xlim[0], right=xlim[1])
    plt.ylim(top=ylim[0], bottom=ylim[1])
    ax = fig.add_subplot(111)
    pairs = plt.ginput(n=1000, timeout=0, mouse_add=1, mouse_pop=2, mouse_stop=3)

    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    figname, _ = os.path.splitext(outputfilename)
    figname += ".png"
    dirname = "/".join(figname.split("/")[:-1])
    os.makedirs(dirname, exist_ok=True)

    fig = plt.figure()
    plt.xlim(left=xlim[0], right=xlim[1])
    plt.ylim(top=ylim[0], bottom=ylim[1])
    plt.plot(x, y, "o")
    plt.savefig(figname)

    with open(outputfilename, "w") as f:
        for p in pairs:
            f.write("%.10f %.10f\n" % p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make dataset for implicit curve approximation.")
    parser.add_argument("outputfilename", type=str, help="path to output file (txt)")
    args = parser.parse_args()

    make_dataset(args.outputfilename)
