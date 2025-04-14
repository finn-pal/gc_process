import json

import gc_utils  # type: ignore
import numpy as np
from astropy.io import ascii
from tqdm import tqdm


def get_accretion(halt, sim: str, halo_tid: int, tid_main_lst: list, sim_dir: str, t_dis: float = -1) -> dict:
    """
        Find if the gc has been accreted or formed in-situ. If accreted find details of its accretion.

    Args:
        halt (_type_): Halo tree
        halo_tid (int): Halo tree halo id
        tid_main_lst (list): List of halo tree halo ids (tid) tracing the main progenitors of the most massive
            galaxy at z = 0.
        fire_dir (str): Directory of the FIRE simulation data (of form "/m12i_res7100")
        t_dis (_type_): If applicabale, this is the time of gc disruption.

    Returns:
        dict:
            0 -> Time of accretion (Time when gc is now assigned to a most massive progenitor of the main
                galaxy)
            1 -> Halo tree halo id of the gc at the time of accretion
            2 -> Halo catalogue halo id of the gc at the time of accretion
            3 -> Snapshot at the time of accretion
            4 -> Halo tree halo id of the gc at the snapshot before accretion
            5 -> Halo catalogue halo id of the gc at the snapshot before accretion
            6 -> Snapshot at the snapshot before accretion
            7 -> Survived accretion flag. If gc is disrupted at accretion set to 0. If discrupted before
                accretion or if not relevant set to -1 otherwise if survived accretion set to 1
    """
    # snapshot times file
    snapshot_times = "/snapshot_times.txt"

    # if gc is not accreted then list acc_flag as 0 and other all values as -1
    # I don't think there are any instances of this (by structure of the GC model)
    if halo_tid in tid_main_lst:
        t_acc = -1
        halo_acc_tid = -1
        halo_acc_cid = -1
        snap_acc = -1
        halo_pre_acc_tid = -1
        halo_pre_acc_cid = -1
        snap_pre_acc = -1
        acc_survive = -1

    else:
        # open table of snapshot information
        fire_dir = sim_dir + sim + "/" + sim + "_res7100/"
        with open(fire_dir + snapshot_times) as f:
            content = f.readlines()
            content = content[5:]
        snap_all = ascii.read(content)

        # get list of descendents (tid's) of the halo
        desc_lst = gc_utils.get_descendants_halt(halo_tid, halt)

        # find which descendent of the halo of formation has been accreted into the main galaxy
        idx_lst = np.array([1 if halt["tid"][idx] in tid_main_lst else 0 for idx in desc_lst])
        idx_acc = np.where(idx_lst == 1)[0][0]

        # get the time of accretion
        snap_acc = halt["snapshot"][desc_lst[idx_acc]]
        t_acc = snap_all["time[Gyr]"][snap_acc]
        t_acc = np.round(t_acc, 3)

        # if gc is not disrupted or disrupted after accretion then get all details of accretion
        if (t_dis == -1) or (t_dis > t_acc):
            idx_pre_acc = idx_acc - 1

            t_acc = t_acc
            halo_acc_tid = halt["tid"][desc_lst[idx_acc]]
            halo_acc_cid, snap_acc = gc_utils.get_halo_cid(halt, halo_acc_tid, fire_dir)
            halo_pre_acc_tid = halt["tid"][desc_lst[idx_pre_acc]]
            halo_pre_acc_cid, snap_pre_acc = gc_utils.get_halo_cid(halt, halo_pre_acc_tid, fire_dir)
            acc_survive = 1  # survived

        # if gc disrupted at the time of accretion then get all details of accretion
        # I don't believe there are any instances of this
        elif t_dis == t_acc:
            idx_pre_acc = idx_acc - 1

            t_acc = t_acc
            halo_acc_tid = halt["tid"][desc_lst[idx_acc]]
            halo_acc_cid, snap_acc = gc_utils.get_halo_cid(halt, halo_acc_tid, fire_dir)
            halo_pre_acc_tid = halt["tid"][desc_lst[idx_pre_acc]]
            halo_pre_acc_cid, snap_pre_acc = gc_utils.get_halo_cid(halt, halo_pre_acc_tid, fire_dir)
            acc_survive = 0  # did not survived

        # if gc disrupted before halo is accreted then set all values to -1
        else:
            ### UPDATE (11/03/2025) commented out lines below is old
            idx_pre_acc = idx_acc - 1
            # might later combine with the above (t_dis <= t_acc)

            # t_acc = -1
            # halo_acc_tid = -1
            # halo_acc_cid = -1
            # snap_acc = -1
            # halo_pre_acc_tid = -1
            # halo_pre_acc_cid = -1
            # snap_pre_acc = -1

            ### UPDATE (12/03/2025) commented out lines below is old
            # t_acc = -1

            # so I can still make ex-situ mass plots. This relates to when they would have been accreted
            # if had survived.
            t_acc = t_acc

            # these values hold no meaning as not accreted, but are required to get group_id
            halo_acc_tid = halt["tid"][desc_lst[idx_acc]]
            halo_acc_cid, snap_acc = gc_utils.get_halo_cid(halt, halo_acc_tid, fire_dir)
            halo_pre_acc_tid = halt["tid"][desc_lst[idx_pre_acc]]
            halo_pre_acc_cid, snap_pre_acc = gc_utils.get_halo_cid(halt, halo_pre_acc_tid, fire_dir)

            ### UPDATE (11/03/2025) commented out line below is old
            # acc_survive = -1
            acc_survive = 0

    # need the int conversions to make json output work
    halo_acc_tid = np.array(halo_acc_tid, dtype=int).tolist()
    halo_acc_cid = np.array(halo_acc_cid, dtype=int).tolist()
    snap_acc = np.array(snap_acc, dtype=int).tolist()
    halo_pre_acc_tid = np.array(halo_pre_acc_tid, dtype=int).tolist()
    halo_pre_acc_cid = np.array(halo_pre_acc_cid, dtype=int).tolist()
    snap_pre_acc = np.array(snap_pre_acc, dtype=int).tolist()

    accretion_dict = {
        "accretion_time": t_acc,
        "accretion_halo_tid": halo_acc_tid,
        "accretion_halo_cid": halo_acc_cid,
        "accretion_snapshot": snap_acc,
        "pre_accretion_halo_tid": halo_pre_acc_tid,
        "pre_accretion_halo_cid": halo_pre_acc_cid,
        "pre_accretion_snapshot": snap_pre_acc,
        "survived_accretion": acc_survive,
    }

    return accretion_dict


def group_accretion(
    accretion_flag: list[int],
    pre_accretion_halo_tid: list[int],
    analyse_flag: list[int],
    survived_past_accretion: list[int],
) -> list:
    """
    Group accretion's together for easy identification. Group 0 is in-situ formation, -1 is gc's disrupted
    before accretion and all other values relate to the halo tid of the gc the snapshot before accretion.

    Args:
        accretion_flag (list[int]): Accretion flag (0 for in-situ, 1 for accreted).
        pre_accretion_halo_tid (list[int]): Halo of GC in snapshot before it is accreted.
        analyse_flag (list[int]): Flag to check whether to analyse or not (0 is skip, 1 is to analyse).

    Returns:
        list[int]: list of group id's to be added to data tables
    """
    group_id_lst = []

    for pre_acc_tid, accr_flag, an_flag, surv_flag in zip(
        pre_accretion_halo_tid, accretion_flag, analyse_flag, survived_past_accretion
    ):
        if an_flag == 0:
            group_id_lst.append(-2)
            continue

        if accr_flag == 0:  # formed in-situ
            group_id_lst.append(0)
            continue

        if surv_flag == 0:  # gc destroyed before / during accretion
            ### UPDATE (11/03/2025) commented out line below is old
            # group_id_lst.append(-1)
            group_id_lst.append(-pre_acc_tid)  # add a (-) to ensure detactable as died before
            continue

        else:
            group_id_lst.append(pre_acc_tid)

    # need this to make json output work
    group_id_lst = [int(group_id) for group_id in group_id_lst]

    return group_id_lst


def process_data(
    sim: str,
    it: int,
    sim_dir: str,
    main_halo_tid: int,
    halt,
    real_flag=1,
    survive_flag=None,
    accretion_flag=None,
):
    """
    Process interim data and add additional information necessary for analysis. This includes deriving
    accretion information about the gc particles. There is also the option to filter out based on flags.

    Args:
        sim (str): Simulation of interest (of form "m12i").
        it (int): Iteration number. This realtes to the randomiser seed used in the gc model.
        sim_dir (str): Directory of the simulation data.
        main_halo_tid (int): Mian halo tree halo id
        real_flag (int, optional): 0 means not real (see convert_data function for details). 1 means real.
            None means to include both. Defaults to 1.
        survive_flag (_type_, optional): 0 means has not survived. 1 means has survived. None means to include
            both. Defaults to None.
        accretion_flag (_type_, optional): 0 means has not been accreted. 1 means has been accreted. None
            means to include both. Defaults to None.
    """

    # get list of main progenitors of most massive galaxy across all redshifts
    tid_main_lst = gc_utils.main_prog_halt(halt, main_halo_tid)

    # get group from iteration (it)
    it_id = gc_utils.iteration_name(it)

    # data_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    data_file = sim_dir + sim + "/gc_results/interim/" + it_id + ".json"

    # int_data = h5py.File(data_file, "r")  # open interim data file

    with open(data_file, "r") as json_file:
        int_data = json.load(json_file)

    # relevant list of flags
    real_flag_lst = int_data[it_id]["source"]["real_flag"]
    surv_flag_lst = int_data[it_id]["source"]["survive_flag"]

    # if gc was formed in a halo that is a main projenitor of the main halo at z = 0 then it was not accreted
    # 1 is accreted 0 is not accreted

    accr_flag_lst = [1 if mpb == 0 else 0 for mpb in int_data[it_id]["source"]["is_mpb"]]

    analyse_flag = []  # based on flags in function variables (0 = skip, 1 = analyse)

    # based on flags this for loop determines which GCs to analyse and which to skip
    for r_flag, s_flag, a_flag in zip(real_flag_lst, surv_flag_lst, accr_flag_lst):
        if real_flag is not None:
            if r_flag != real_flag:
                analyse_flag.append(0)
                continue

        if survive_flag is not None:
            if s_flag != survive_flag:
                analyse_flag.append(0)
                continue

        if accretion_flag is not None:
            if a_flag != accretion_flag:
                analyse_flag.append(0)
                continue

        analyse_flag.append(1)

    halo_zform = int_data[it_id]["source"]["halo_zform"]
    t_dis = int_data[it_id]["source"]["t_dis"]

    # empty lists to be filled
    t_acc_lst = []
    halo_acc_tid_lst = []
    halo_acc_cid_lst = []
    snap_acc_lst = []
    halo_pre_acc_tid_lst = []
    halo_pre_acc_cid_lst = []
    snap_pre_acc_lst = []
    acc_survive_lst = []

    # if analyse flag is 0 then give the following values as nan
    accretion_dict_skip = {
        "accretion_time": -2,
        "accretion_halo_tid": -2,
        "accretion_halo_cid": -2,
        "accretion_snapshot": -2,
        "pre_accretion_halo_tid": -2,
        "pre_accretion_halo_cid": -2,
        "pre_accretion_snapshot": -2,
        "survived_accretion": -2,
    }

    # get accretion information where relevant
    for h_form, t_d, an_flag in tqdm(
        zip(halo_zform, t_dis, analyse_flag),
        ncols=150,
        total=len(analyse_flag),
        desc=it_id + " Processing Data....................",
    ):
        # is analysis flag = 0 then set to np.nan and skip
        # if an_flag == 0:

        if an_flag == 0:
            accretion_dict = accretion_dict_skip

        else:
            gc_utils.block_print()
            accretion_dict = get_accretion(halt, sim, h_form, tid_main_lst, sim_dir, t_d)
            gc_utils.enable_print()

        t_acc_lst.append(accretion_dict["accretion_time"])
        halo_acc_tid_lst.append(accretion_dict["accretion_halo_tid"])
        halo_acc_cid_lst.append(accretion_dict["accretion_halo_cid"])
        snap_acc_lst.append(accretion_dict["accretion_snapshot"])
        halo_pre_acc_tid_lst.append(accretion_dict["pre_accretion_halo_tid"])
        halo_pre_acc_cid_lst.append(accretion_dict["pre_accretion_halo_cid"])
        snap_pre_acc_lst.append(accretion_dict["pre_accretion_snapshot"])
        acc_survive_lst.append(accretion_dict["survived_accretion"])

    ptype_lst = []
    for qual in int_data[it_id]["source"]["quality"]:
        ptype = gc_utils.particle_type(qual)
        ptype_lst.append(ptype)

    # add accretion group
    group_id_lst = group_accretion(accr_flag_lst, halo_pre_acc_tid_lst, analyse_flag, acc_survive_lst)
    # int_data.close()

    it_dict = {
        "accretion_flag": accr_flag_lst,
        "analyse_flag": analyse_flag,
        "t_acc": t_acc_lst,
        "halo_acc_tid": halo_acc_tid_lst,
        "halo_acc_cid": halo_acc_cid_lst,
        "snap_acc": snap_acc_lst,
        "halo_pre_acc_tid": halo_pre_acc_tid_lst,
        "halo_pre_acc_cid": halo_pre_acc_cid_lst,
        "snap_pre_acc": snap_pre_acc_lst,
        "survived_accretion": acc_survive_lst,
        "ptype": ptype_lst,
        "group_id": group_id_lst,
    }

    for key in it_dict.keys():
        updt_dict = {key: it_dict[key]}
        int_data[it_id]["source"].update(updt_dict)

    with open(data_file, "w") as json_file:
        json.dump(int_data, json_file)
