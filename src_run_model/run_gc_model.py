import argparse

from tools.gc_model import prep_gc_model, run_gc_model


def main(sim: str, it: int, location: str):
    params = prep_gc_model(sim, it, location)
    run_gc_model(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulation", required=True, type=str, help="simulation name (e.g. m12i)")
    parser.add_argument("-i", "--iteration", required=True, type=int, help="random seed for gc formation")
    parser.add_argument("-l", "--location", required=True, type=str, help="either local or katana")
    args = parser.parse_args()

    main(args.simulation, args.iteration, args.location)
