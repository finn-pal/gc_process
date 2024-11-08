import argparse
import json
import os
import sys

import h5py
import numpy as np
from tools.utils import iteration_name


def main(simulation: str, iteration_low_limit: int, iteration_up_limit: int, location: str):
    if location == "local":
        sim_dir = "/Users/z5114326/Documents/simulations/"
        data_dir = "/Users/z5114326/Documents/GitHub/gc_process_katana/data/"
        sim_codes = "data/external/simulation_codes.json"

    elif location == "katana":
        data_dir = "/srv/scratch/astro/z5114326/gc_process/data/"
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"
        sim_codes = data_dir + "external/simulation_codes.json"

    else:
        print("Incorrect location provided. Must be local or katana.")
        sys.exit()

    save_file = sim_dir + simulation + "/" + simulation + "_processed.hdf5"  # save location

    if not os.path.exists(save_file):
        h5py.File(save_file, "w")

    with h5py.File(save_file, "a") as hdf:
        for it in range(iteration_low_limit, iteration_up_limit + 1):
            it_id = iteration_name(it)
            data_file = data_dir + "results/" + simulation + "/interim/" + it_id + ".json"
            with open(data_file, "r") as sim_json:
                int_data = json.load(sim_json)
            data_dict = int_data[it_id]["source"]

            if it_id in hdf.keys():
                grouping = hdf[it_id]
            else:
                grouping = hdf.create_group(it_id)
            if "source" in grouping.keys():
                source = grouping["source"]
            else:
                source = grouping.create_group("source")
            for key in data_dict.keys():
                if key in source.keys():
                    del source[key]
                if key == "ptype":
                    source.create_dataset(key, data=data_dict[key])
                else:
                    source.create_dataset(key, data=np.array(data_dict[key]))

    # start adding masses at all available snapshots
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    snap_offset = sim_data[simulation]["offset"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-a", "--iteration_low_limit", required=True, type=int, help="lower bound iteration")
    parser.add_argument("-b", "--iteration_up_limit", required=True, type=int, help="upper bound iteration")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    args = parser.parse_args()

    main(args.simulation, args.iteration_low_limit, args.iteration_up_limit, args.location)
