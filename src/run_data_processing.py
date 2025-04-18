import argparse
import json
import sys

import gc_utils  # type: ignore

# from gc_utils import get_halo_tree  # type: ignore
from tools.convert_data import convert_data
from tools.process_data import process_data


def main(simulation: str, iteration: int, location: str, real_flag=1, survive_flag=None, accretion_flag=None):
    if location == "local":
        sim_dir = "../../simulations/"
        # data_dir = "data/"
        # sim_codes = data_dir + "external/simulation_codes.json"

    elif location == "katana":
        # data_dir = "/srv/scratch/astro/z5114326/gc_process/data/"
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"
        # sim_codes = data_dir + "external/simulation_codes.json"

    else:
        print("Incorrect location provided. Must be local or katana.")
        sys.exit()

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as json_file:
        sim_data = json.load(json_file)

    # with open(sim_codes) as json_file:
    #     data = json.load(json_file)

    offset = sim_data[simulation]["offset"]
    main_halo_tid = [sim_data[simulation]["halo"]]

    halt = gc_utils.get_halo_tree(simulation, sim_dir)
    convert_data(simulation, iteration, offset, sim_dir)
    process_data(simulation, iteration, sim_dir, main_halo_tid, halt, real_flag, survive_flag, accretion_flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-i", "--iteration", required=True, type=int, help="random seed for gc formation")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")

    # processing flags: (1 for real, 0 for not real, None for both)
    parser.add_argument("-r", "--real_flag", required=False, help="is the gc real.")
    parser.add_argument("-u", "--survive_flag", required=False, help="has the gc survived to redshift 0.")
    parser.add_argument("-a", "--accretion_flag", required=False, help="has the gc been accreted?")

    args = parser.parse_args()

    arg_dict = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}

    flag_list = ["real_flag", "survive_flag", "accretion_flag"]
    flag_val_list = ["0", "1", "None"]

    for flag in flag_list:
        if flag in arg_dict.keys():
            if arg_dict[flag] not in flag_val_list:
                print("Incorrect flag value provided. Must be 0, 1 or None.")
                sys.exit()
            elif arg_dict[flag] == "None":
                arg_dict[flag] = None
            else:
                arg_dict[flag] = int(arg_dict[flag])

    main(**arg_dict)
