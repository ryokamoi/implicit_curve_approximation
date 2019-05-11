import os
import shutil
import pickle as pkl
from datetime import datetime
import argparse

import numpy as np

from utils.tools import read_dataset, visualize_implicit
from utils.params import read_config
from model import *


def main(configfile: str) -> None:
    params = read_config(configfile)

    # output directory
    now = datetime.now().strftime("%y%m%d-%H%M")
    datasetname, _ = os.path.splitext(params.dataset)
    datasetname = datasetname.split("/")[-1]
    dirname = "output/" + now + "-%s-%s" % (params.model, datasetname)
    os.makedirs(dirname)
    shutil.copy(configfile, dirname + "/config.txt")

    # get result
    zeroset = read_dataset(params.dataset + "/zeroset.txt")
    model = eval(params.model)
    if not params.threeL:
        func = model(zeroset, params)
    else:
        inner = read_dataset(params.dataset + "/inner.txt")
        outer = read_dataset(params.dataset + "/outer.txt")
        func = model(zeroset, params, outer=outer, inner=inner)

    # save result
    visualize_implicit(func, filename=dirname + "/implicit.png", dataset=zeroset)

    with open(dirname + "/function.pkl", "wb") as f:
        pkl.dump(func, f)


if __name__ == "__main__":
    np.random.seed(19951202)

    parser = argparse.ArgumentParser(description="Implicit curve approximation.")
    parser.add_argument("configfile", type=str, help="path to config file")
    args = parser.parse_args()

    main(args.configfile)
