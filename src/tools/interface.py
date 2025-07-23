import os
from multiprocessing import Pool

import h5py
import numpy as np
import yt

# yt.funcs.mylog.setLevel(50)  # ignore yt's output

# Note: Mh is DM halo mass, not total mass!
# Note: merger tree snap and sim snap differ! All convert to merger tree snap

full_snap = np.array(
    [
        20,
        23,
        26,
        29,
        33,
        37,
        41,
        46,
        52,
        59,
        67,
        77,
        88,
        102,
        120,
        142,
        172,
        214,
        277,
        294,
        312,
        332,
        356,
        382,
        412,
        446,
        486,
        534,
        590,
        591,
        592,
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        600,
    ],
    dtype=int,
)

# full_snap = np.array([20, 23, 26, 29, 33, 37, 41, 46, 52, 59, 67, 77, 88, 95, 102, 111, 120, 127, 134, 142, 150,
#     158, 165, 172, 179, 186, 193, 200, 207, 214, 221, 228, 235, 242, 249, 256, 277, 263, 270, 294, 312, 332, 356,
#     382, 412, 446, 486, 534, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600], dtype=int)

Ob = 0.0455
Om = 0.272

fb = Ob / Om  # baryon fraction to correct for halo mass


def yt_load_all(path):
    d = []
    for snap in full_snap:
        d.append(yt.load(path + "snapdir_%03d" % snap))
    return d


def yt_load_all_v2(path):
    d = []
    for snap in full_snap:
        file_path = os.path.join(path, f"snapdir_{snap:03d}", f"snapshot_{snap:03d}.0.hdf5")
        d.append(yt.load(file_path))
    return d


def get_fp(tree):
    subid = tree[:, 1].astype(int)
    descid = tree[:, 3].astype(int)
    is_fp = tree[:, 14].astype(bool)
    fp = -1 * np.ones(len(tree), dtype=int)

    for i in range(len(tree)):
        if is_fp[i] and descid[i] > 0:
            idx = np.where(subid == descid[i])[0]
            fp[idx[0]] = subid[i]
            print("%d out of %d" % (i, len(tree)))

    return fp


def handle_merger_tree(tree_base, save_base, hid, skip=48):
    print("Have begun merger tree")
    fields = [
        "SubhaloMass",
        "FirstProgenitorID",
        "SubhaloID",
        "SnapNum",
        "MainLeafProgenitorID",
        "NextProgenitorID",
        "DescendantID",
        "SubfindID",
        "SubhaloPos",
        "SubhaloVel",
        "ScaleRad",
        "J",
    ]
    field_idxs = [10, -1, 1, 31, 34, -1, 3, 1, 17, 20, 12, 23]
    dtypes = ["f8", "i8", "i8", "i8", "i8", "i8", "i8", "i8", "f8", "f8", "f8", "f8"]

    tree_path = tree_base
    tree_full = np.loadtxt(tree_path, skiprows=skip)
    tree = tree_full[tree_full[:, 29] == hid]

    print("making merger tree file")
    filename = save_base + "merger_tree_%d.hdf5" % hid
    if os.path.exists(filename):
        os.remove(filename)

    with h5py.File(filename, "w") as f:
        for field, field_idx, dtype in zip(fields, field_idxs, dtypes):
            if field == "SubhaloMass":
                f.create_dataset(field, data=tree[:, field_idx] / 1e10 / (1 - fb), dtype=dtype)
            elif field == "FirstProgenitorID":
                f.create_dataset(field, data=get_fp(tree), dtype=dtype)
            elif field == "NextProgenitorID":
                f.create_dataset(field, data=np.zeros(len(tree)), dtype=dtype)  # not used
            elif field == "SubhaloPos":
                pos = np.array(
                    [tree[:, field_idx] * 1e3, tree[:, field_idx + 1] * 1e3, tree[:, field_idx + 2] * 1e3]
                ).T
                f.create_dataset(field, data=pos, dtype=dtype)
            elif field == "SubhaloVel":
                vel = np.array([tree[:, field_idx], tree[:, field_idx + 1], tree[:, field_idx + 2]]).T
                f.create_dataset(field, data=vel, dtype=dtype)
            elif field == "J":
                pos = np.array([tree[:, field_idx], tree[:, field_idx + 1], tree[:, field_idx + 2]]).T
                f.create_dataset(field, data=pos, dtype=dtype)
            else:
                f.create_dataset(field, data=tree[:, field_idx], dtype=dtype)

    print("handle_merger_tree hid: %d done" % hid)


def handle_halo(sim_base, save_base, hid, skip=48, mh_min=1e8, offset=4):
    print("Have begun part handle")
    full_snap_rockstar = full_snap - offset

    parttypes = ["dm", "stars", "gas"]
    fields_list = [
        ["Coordinates", "Velocities", "ParticleIDs", "Potential", "Masses"],
        ["Coordinates", "Velocities", "ParticleIDs", "GFM_StellarFormationTime", "Potential", "Masses"],
        ["Coordinates", "Velocities", "ParticleIDs", "Potential", "Masses"],
    ]
    yt_field_list = [
        [
            ("PartType1", "Coordinates"),
            ("PartType1", "Velocities"),
            ("PartType1", "ParticleIDs"),
            ("PartType1", "Potential"),
            ("PartType1", "Masses"),
        ],
        [
            ("PartType4", "Coordinates"),
            ("PartType4", "Velocities"),
            ("PartType4", "ParticleIDs"),
            ("PartType4", "StellarFormationTime"),
            ("PartType4", "Potential"),
            ("PartType4", "Masses"),
        ],
        [
            ("PartType0", "Coordinates"),
            ("PartType0", "Velocities"),
            ("PartType0", "ParticleIDs"),
            ("PartType0", "Potential"),
            ("PartType0", "Masses"),
        ],
    ]
    dtypes_list = [
        ["f8", "f8", "i8", "f8", "f8"],
        ["f8", "f8", "i8", "f8", "f8", "f8"],
        ["f8", "f8", "i8", "f8", "f8"],
    ]
    units_list = [
        ["kpccm/h", "km/s", None, None, "Msun"],
        ["kpccm/h", "km/s", None, None, None, "Msun"],
        ["kpccm/h", "km/s", None, None, "Msun"],
    ]

    # dataset = yt_load_all(sim_base)
    dataset = yt_load_all_v2(sim_base)

    tree_path = tree_base
    tree_full = np.loadtxt(tree_path, skiprows=skip)
    tree = tree_full[tree_full[:, 29] == hid]
    ntot = len(tree)

    print("making part file")
    filename = save_base + "halo_%d.hdf5" % hid
    if not os.path.exists(filename):
        with h5py.File(filename, "w") as f:
            f.attrs["completed"] = False

    with h5py.File(filename, "a") as f:
        f.attrs["completed"] = False
        if not f.attrs["completed"]:
            for n, (h, s, m) in enumerate(
                zip(tree[:, 1], tree[:, 31], tree[:, 10])
            ):  # s is the merger tree snap
                if not s in full_snap_rockstar:
                    continue
                if m < mh_min * 0.5:  # skip small halos
                    continue

                print(n, ntot)
                if n % 100 == 0:
                    print(n, ntot)

                if not "snap_%d_halo_%d" % (s, h) in f:
                    d = f.create_group("snap_%d_halo_%d" % (s, h))
                    d.attrs["completed"] = False

                d = f["snap_%d_halo_%d" % (s, h)]

                if d.attrs["completed"] and n != 0:
                    print("snap_%d_halo_%d already done" % (s, h))
                    continue

                idx_in_tree = np.where(np.abs(tree[:, 1] - h) < 0.5)[0]
                assert len(idx_in_tree) == 1
                halo_tree = tree[idx_in_tree[0]]

                idx = np.where(full_snap_rockstar == s)[0][
                    0
                ]  # 4 is the offset. rockstar didn't find any halo before snap 4 (just m12i)

                ds = dataset[idx]
                hpos = halo_tree[17:20] * ds.arr(1, "Mpccm/h")  # comoving Mpc/h
                r = np.max([halo_tree[11], 10, halo_tree[12]]) * ds.arr(1, "kpccm/h")

                cutout = ds.sphere(hpos, r)

                for parttype, fields, yt_fileds, dtypes, units in zip(
                    parttypes, fields_list, yt_field_list, dtypes_list, units_list
                ):
                    d2 = d.create_group(parttype)

                    d2.attrs["count"] = len(cutout[yt_fileds[0]])
                    for field, yt_field, dtype, unit in zip(fields, yt_fileds, dtypes, units):
                        if unit is None:
                            d2.create_dataset(field, data=cutout[yt_field], dtype=dtype)
                        else:
                            d2.create_dataset(field, data=cutout[yt_field].to(unit).value, dtype=dtype)

                d.attrs["completed"] = True

            f.attrs["completed"] = True

    print("handle_halo hid: %d done" % hid)


if __name__ == "__main__":
    # number of processor
    Np = 2

    # base path for FIRE m12i # 596
    # skip = 50
    # offset = 4
    # tree_base = "/Volumes/My Passport for Mac/m12i_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    # sim_base = "/Volumes/My Passport for Mac/m12i_res7100/output/"
    # save_base = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/interface_output/"
    # save_tree = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/base_tree/"
    # save_halo = "/Users/z5114326/Documents/GitHub/GC_kinematics/data/base_halo/"
    # hid = 25236877

    # # base path for FIRE m12c # 599
    skip = 50
    offset = 1
    # tree_base = "/nfs/astro2/ybchen/FIRE/m12c_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    tree_base = (
        "/Volumes/One Touch/simulations/m12c/m12c_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    )
    # sim_base = "/nfs/astro2/ybchen/FIRE/m12c_res7100/output/"
    sim_base = "/Volumes/One Touch/simulations/m12c/m12c_res7100/output/"
    # save_base = "/nfs/astro2/ybchen/fire_halos/"
    save_base = "/Volumes/One Touch/simulations/m12c/interface_output/"
    hid = 89423951

    # # base path for FIRE m12r # 599
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12r_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12r_res7100/output/'
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # hid = 36334767

    # # base path for FIRE m12m
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12m_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # tree_base = (
    #     "/Volumes/One Touch/simulations/m12m/m12m_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    # )
    # sim_base = "/Volumes/One Touch/simulations/m12m/m12m_res7100/output/"
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12m_res7100/output/'
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # save_base = "/Volumes/One Touch/simulations/m12m/interface_output/"
    # hid = 79502363

    # # base path for FIRE m12f
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12f_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # tree_base = (
    #     "/Volumes/One Touch/simulations/m12f/m12f_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    # )
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12f_res7100/output/'
    # sim_base = "/Volumes/One Touch/simulations/m12f/m12f_res7100/output/"
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # save_base = "/Volumes/One Touch/simulations/m12f/interface_output/"
    # hid = 53854632

    # # base path for FIRE m12b # 599
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12b_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # tree_base = (
    #     "/Volumes/One Touch/simulations/m12b/m12b_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat"
    # )
    # # sim_base = '/nfs/astro2/ybchen/FIRE/m12b_res7100/output/'
    # sim_base = "/Volumes/One Touch/simulations/m12b/m12b_res7100/output/"
    # # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # save_base = "/Volumes/One Touch/simulations/m12b/interface_output/"
    # hid = 39810368

    # # base path for FIRE m12w # 599
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12w_res7100/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12w_res7100/output/'
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # hid = 42522760

    # # base path for FIRE m12z # ???
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12w_res4200/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12w_res4200/output/'
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # hid = ?????

    # # base path for FIRE R&J # 599
    # skip = 50
    # offset = 1
    # tree_base = '/nfs/astro2/ybchen/FIRE/m12_elvis_RomeoJuliet_res3500/halo/rockstar_dm/catalog/trees/tree_0_0_0.dat'
    # sim_base = '/nfs/astro2/ybchen/FIRE/m12_elvis_RomeoJuliet_res3500/output/'
    # save_base = '/nfs/astro2/ybchen/fire_halos/'
    # subs = [140611946, 140517229]
    # hid = 140611946

    # para_list = []
    # for hid in subs:
    #     para_list.append((tree_base, save_base, hid, skip))

    # with Pool(Np) as p:
    #     p.starmap(handle_merger_tree, para_list)

    # para_list = []
    # for hid in subs:
    #     para_list.append((sim_base, save_base, hid, skip, 1e8, offset))

    # with Pool(Np) as p:
    #     p.starmap(handle_halo, para_list)

    #### USE CORRECT yt_load_all_v2
    handle_merger_tree(tree_base, save_base, hid, skip)
    handle_halo(sim_base, save_base, hid, skip, offset=offset)
