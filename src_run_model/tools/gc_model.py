import json
import os
import sys

import numpy as np
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
        sim_codes = "data/external/simulation_codes.json"
        resultpath = "data/results/" + sim + "/raw/" + "it_%d/" % it
        sim_path = "data/simulations/" + sim + "/interface_output/"

    elif location == "katana":
        sim_codes = "/srv/scratch/astro/z5114326/gc_process/data/external/simulation_codes.json"
        resultpath = "/srv/scratch/astro/z5114326/gc_process/data/results/" + sim + "/raw/" + "it_%d/" % it
        sim_path = "/srv/scratch/astro/z5114326/simulations/" + sim + "/interface_output/"

    else:
        print("Incorrect location provided. Must be local or katana.")
        sys.exit()

    with open(sim_codes) as json_file:
        data = json.load(json_file)

    subs = [data[sim]["halo"]]
    offset = data[sim]["offset"]
    h100 = data[sim]["h100"]
    Ob = data[sim]["Ob"]
    Om = data[sim]["Om"]

    if location == "local":
        redshift_snap = np.loadtxt("data/external/fire_z_list.txt", dtype=float)[offset:]

    elif location == "katana":
        redshift_snap = np.loadtxt(
            "/srv/scratch/astro/z5114326/gc_process/data/external/fire_z_list.txt", dtype=float
        )[offset:]

    params["resultspath"] = resultpath
    params["seed"] = int(it)
    params["subs"] = subs
    params["redshift_snap"] = redshift_snap
    params["full_snap"] = params["full_snap"] - offset
    params["analyse_snap"] = params["full_snap"][-1]
    params["h100"] = h100
    params["Ob"] = Ob
    params["Om"] = Om
    params["base_tree"] = sim_path
    params["base_halo"] = sim_path

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
    evolve(run_params, return_t_disrupt=True)

    if params["verbose"]:
        print("\nModel was run on %d halo(s).\n" % len(params["subs"]))
