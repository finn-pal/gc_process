import time

import gc_utils  # type: ignore
import h5py
import numpy as np
import utilities as ut


def get_basic_kinematics(
    part,
    sim: str,
    it_lst: list[int],
    snapshot: int,
    sim_dir: str,
    data_dict: dict = {},
    host_index: int = 0,
):
    snap_id = gc_utils.snapshot_name(snapshot)

    proc_file = sim_dir + sim + "/" + sim + "_processed.hdf5"
    proc_data = h5py.File(proc_file, "r")  # open processed data file

    it_dict = {}
    start = time.time()
    for it in it_lst:
        print(it)
        it_id = gc_utils.iteration_name(it)

        gc_id_snap = proc_data[it_id]["snapshots"][snap_id]["gc_id"]

        if len(gc_id_snap) is None:
            continue

        ptype_byte_snap = proc_data[it_id]["snapshots"][snap_id]["ptype"]
        ptype_snap = [ptype.decode("utf-8") for ptype in ptype_byte_snap]

        host_name = ut.catalog.get_host_name(host_index)

        x_lst = []
        y_lst = []
        z_lst = []
        vx_lst = []
        vy_lst = []
        vz_lst = []

        r_cyl_lst = []
        phi_cyl_lst = []
        vr_cyl_lst = []
        vphi_cyl_lst = []

        init_cond_lst = []

        r_lst = []

        ep_fire_lst = []
        ek_lst = []

        lx_lst = []
        ly_lst = []
        lz_lst = []

        for gc, ptype in zip(gc_id_snap, ptype_snap):
            idx = np.where(part[ptype]["id"] == gc)[0][0]
            pos_xyz = part[ptype].prop(f"{host_name}.distance.principal", idx)
            vel_xyz = part[ptype].prop(f"{host_name}.velocity.principal", idx)

            pos_cyl = part[ptype].prop(f"{host_name}.distance.principal.cylindrical", idx)
            vel_cyl = part[ptype].prop(f"{host_name}.velocity.principal.cylindrical", idx)

            init_cond = np.concatenate((pos_xyz, vel_xyz))

            ep_fir = part[ptype]["potential"][idx]
            ek = 0.5 * np.linalg.norm(vel_xyz) ** 2

            x, y, z = pos_xyz
            vx, vy, vz = vel_xyz

            r_cyl, phi_cyl, _ = pos_cyl
            vr_cyl, vphi_cyl, _ = vel_cyl

            r = np.linalg.norm(pos_xyz)

            lx = y * vz - z * vy
            ly = z * vx - x * vz
            lz = x * vy - y * vx

            x_lst.append(x)
            y_lst.append(y)
            z_lst.append(z)
            vx_lst.append(vx)
            vy_lst.append(vy)
            vz_lst.append(vz)

            r_cyl_lst.append(r_cyl)
            phi_cyl_lst.append(phi_cyl)
            vr_cyl_lst.append(vr_cyl)
            vphi_cyl_lst.append(vphi_cyl)

            r_lst.append(r)

            ep_fire_lst.append(ep_fir)
            ek_lst.append(ek)

            lx_lst.append(lx)
            ly_lst.append(ly)
            lz_lst.append(lz)

            init_cond_lst.append(init_cond)

        kin_dict = {
            "x": np.array(x_lst),
            "y": np.array(y_lst),
            "z": np.array(z_lst),
            "vx": np.array(vx_lst),
            "vy": np.array(vy_lst),
            "vz": np.array(vz_lst),
            "r_cyl": np.array(r_cyl_lst),
            "phi_cyl": np.array(phi_cyl_lst),
            "vr_cyl": np.array(vr_cyl_lst),
            "vphi_cyl": np.array(vphi_cyl_lst),
            "r": np.array(r_lst),
            "ep_fire": np.array(ep_fire_lst),
            "ek": np.array(ek_lst),
            "lx": np.array(lx_lst),
            "ly": np.array(ly_lst),
            "lz": np.array(lz_lst),
        }

        it_dict[it_id] = kin_dict
        del kin_dict

    proc_data.close()

    end = time.time()
    print("time:", end - start)
    data_dict[snap_id] = it_dict

    return data_dict
