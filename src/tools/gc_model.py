import json
import os
import sys

import numpy as np
import pandas as pd
from GC_formation_model import astro_utils, logo
from GC_formation_model.assign import assign
from GC_formation_model.evolve import evolve
from GC_formation_model.form import form
from GC_formation_model.get_tid import get_tid
from GC_formation_model.offset import offset

from tools.params import params

__all__ = ["run_gc_model"]


def prep_gc_model(sim: str, it: int, location: str):
    if location == "local":
        # sim_codes = "data/external/simulation_codes.json"
        # model_snaps = "data/external/model_snapshots.json"
        # resultpath = "data/results/" + sim + "/raw/" + "it_%d/" % it
        sim_dir = "../../simulations/"

    elif location == "katana":
        # sim_codes = "/srv/scratch/astro/z5114326/gc_process/data/external/simulation_codes.json"
        # model_snaps = "/srv/scratch/astro/z5114326/gc_process/data/external/model_snapshots.json"
        # resultpath = "/srv/scratch/astro/z5114326/gc_process/data/results/" + sim + "/raw/" + "it_%d/" % it
        sim_dir = "/srv/scratch/astro/z5114326/simulations/"

    else:
        print("Incorrect location provided. Must be local or katana.")
        sys.exit()

    resultpath = sim_dir + sim + "/gc_results/raw/it_%d/" % it

    redshift_path = sim_dir + sim + "/" + sim + "_res7100/snapshot_times.txt"
    interface_dir = sim_dir + sim + "/interface_output/"

    sim_codes = sim_dir + "simulation_codes.json"
    with open(sim_codes) as sim_json:
        sim_data = json.load(sim_json)

    subs = [sim_data[sim]["halo"]]
    snap_offset = sim_data[sim]["offset"]
    h100 = sim_data[sim]["h100"]
    Ob = sim_data[sim]["Ob"]
    Om = sim_data[sim]["Om"]

    # with open(model_snaps) as snap_json:
    #     snap_data = json.load(snap_json)

    public_snapshot_file = sim_dir + "snapshot_times_public.txt"
    pub_data = pd.read_table(public_snapshot_file, comment="#", header=None, sep=r"\s+")
    pub_data.columns = [
        "index",
        "scale_factor",
        "redshift",
        "time_Gyr",
        "lookback_time_Gyr",
        "time_width_Myr",
    ]
    pub_snaps = np.array(pub_data["index"], dtype=int)

    redshift_snap = np.loadtxt(redshift_path, dtype=float)[:, 2][snap_offset:]

    params["resultspath"] = resultpath
    params["seed"] = int(it)
    params["subs"] = subs
    params["redshift_snap"] = redshift_snap
    params["full_snap"] = pub_snaps - snap_offset
    params["snap_evolve"] = pub_snaps - snap_offset
    params["analyse_snap"] = params["full_snap"][-1]
    params["h100"] = h100
    params["Ob"] = Ob
    params["Om"] = Om
    params["base_tree"] = interface_dir
    params["base_halo"] = interface_dir

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    return params


def run_gc_model(params):
    if params["verbose"]:
        logo.print_logo()
        logo.print_version()
        print("\nWe refer to the following papers for model details:")
        logo.print_papers()
        print("\nRuning model on %d halo(s)." % len(params["subs"]))

    allcat_name = params["allcat_base"] + "_s-%d_p2-%g_p3-%g.txt" % (
        params["seed"],
        params["p2"],
        params["p3"],
    )

    run_params = params
    run_params["allcat_name"] = allcat_name

    run_params["cosmo"] = astro_utils.cosmo(
        h=run_params["h100"], omega_baryon=run_params["Ob"], omega_matter=run_params["Om"]
    )

    form(run_params)
    offset(run_params)
    assign(run_params)
    get_tid(run_params)

    for snap in params["snap_evolve"]:
        evolve(run_params, return_t_disrupt=True, at_snap=snap)

    if params["verbose"]:
        print("\nModel was run on %d halo(s).\n" % len(params["subs"]))
