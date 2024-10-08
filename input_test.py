import argparse


def func(name="Joe", colour="Green"):
    print(f"Hi, my name is {name} and my favourite colour is {colour}.")


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=False)
parser.add_argument("-c", "--colour", required=False)
args = parser.parse_args()

d = {arg: getattr(args, arg) for arg in vars(args) if getattr(args, arg) is not None}

func(**d)
