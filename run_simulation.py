import os
# supress multithreading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from OCP_BIM_LAMMPS_Diff import OCP_BIM_D
from ReadInput import simulation, species
import numpy as np
import sys
from time import sleep
from shutil import copy
from mpi4py import MPI

# read input file name and trajectory number iTraj
InputFile = sys.argv[1]
iTraj = sys.argv[2]
sim = simulation(InputFile)

# create directory with only processor at rank 0
if MPI.COMM_WORLD.rank == 0:
  if not os.path.isdir(sim.dir):
    os.mkdir(sim.dir)

sleep(3)
os.chdir(sim.dir)

# copy files to that directory
if MPI.COMM_WORLD.rank == 0:
  if not os.path.isfile("./"+InputFile):
    copy("../"+InputFile, ".")
  if not os.path.isfile("./OCP_BIM_LAMMPS_Diff.py"):
    copy("../"+"OCP_BIM_LAMMPS_Diff.py", ".")
  if not os.path.isfile("./ReadInput.py"):
    copy("../"+"ReadInput.py", ".")
  if not os.path.isfile("./post_proc_tool.py"):
    copy("../"+"post_proc_tool.py", ".")
sleep(3)

# run simulation code
OCP_BIM_D(sim, iTraj)
