import sys
import re

# print(sys.argv[1])
# try sys.argv[1]:
try:
    FILE = sys.argv[1]
except IndexError:
    FILE = "../conf.md"

with open(FILE, "r") as f:
    lines = f.readlines()

# import ipdb; ipdb.set_trace()
lines = [l.replace("\t", " | ") for l in lines]


with open(FILE, "w") as f:
    for l in lines:
        f.write(l)