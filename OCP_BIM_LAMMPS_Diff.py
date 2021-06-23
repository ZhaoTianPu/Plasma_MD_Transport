#-------------------------------------------------------------------
#
#           OCP_BIM_Diff_PyL_1.0/OCP_BIM_LAMMPS_Diff.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# This python module contains python functions that perform LAMMPS
# simulation for binary ionic mixture (BIM) via Python/LAMMPS 
# interface wrapper class PyLammps [1], calculate autocorrelation
# functions (ACF) for particle current, individual particle velocity  
# and then evaluate integrals with respect to time to obtain self-  
# and mutual-diffusion using Green-Kubo formula. 
# The units used in here are in dimensionless LJ units. For units 
# and expressions of plasma-related variables and parameter units, 
# please see [6], which is based on [10], [11], [12] and [13].
#
# Glossaries:
# trajectory: a trajectory is one complete simulation, consists of
# an equilibriation stage and a production stage, in [15], the same
# thing is called 'time series'.
# length of ACF: to differentiate from a 'trajectory', the length of 
# an ACF simply refers to the time or timesteps that the ACFs are
# evaluated up to. An ACF which has a length 150 tau means this ACF
# is calculated from its origin up to 150 tau.
# time origin: a time origin refers to the stmulation step when an 
# ACF calculation starts. In general, ACF is defined as 
# ACF(t) = j(0)*j(t) for a time-dependent varable j. At the time origin,
# ACF(0) = j(0)*j(0).
# on the fly: when describing the action of `fix ave/correlate`, 
# the on the fly average over multiple time origin means that the 
# averages of ACFs are evaluated for every specified interval, while
# the simulation still proceeds.
# 
# Module dependencies:
# The versions below are the ones that I use when developing the 
# code. Earlier/later version may or may not work. 
#
# python: 3.8.5, with packages
# - numpy: 1.19 mainly for array manipulations
# - mpi4py: 3.0.3 for LAMMPS MPI supports
# and files
# - ReadInput: for extracting simulation inputs from the simulation class
# - lammps.py: the python codes for the Python/LAMMPS interface.
# Necessary steps for setups before using this module can be found
# in [4], in particular building LAMMPS as a shared library is required
# (which is automatically done if lammps is installed via anaconda3
# [5]).
#
# LAMMPS: 30 Jun 2020. note that there are significant changes in 
# LAMMPS, especially in variable command, over the recent years.
# On top of the release version, when building LAMMPS, the following
# packages are needed, in addition to the basic packages that enables
# LAMMPS to run with MPI:
# -KSPACE: long-range solvers with long-range potentials require this
# package
# -USER-POLY: by Simran Chowdhry, which includes a pair style "poly"
# that enable one to define a inverse-powered polynomial potential for 
# pairs with a potential cutoff [9]. To use pairstyle poly, if USER-POLY 
# or the poly module is not yet included in the LAMMPS official release, 
# then LAMMPS must be built manually [16].  
#
# Here are brief descriptions for the functions in this module. For 
# details about inputs and outputs, please find at the beginning of
# each function, or check them out by .__doc__ or help().
#
# OCP_BIM_D:
# Simulate either one component fluids and binary mixtures, either with 
# specified poly potential or Coulombic potential, and calculate VACF,
# its integral, and particle flux ACF.
#
#-------------------------------------------------------------------
#
# [1]: https://lammps.sandia.gov/doc/Howto_pylammps.html
# [2]: https://lammps.sandia.gov/threads/msg64626.html
# [3]: https://lammps.sandia.gov/doc/compute_vacf.html
# [4]: https://lammps.sandia.gov/doc/Python_head.html
# [5]: https://lammps.sandia.gov/doc/Install_conda.html
# [6]: unit_conversions.pdf
# [7]: https://lammps.sandia.gov/doc/fix_nh.html
# [8]: https://lammps.sandia.gov/doc/fix_ave_time.html
# [9]: folder `USER-POLY`
# [10]: https://lammps.sandia.gov/doc/units.html
# [11]: Daligault PRL 108 225004 (2012)
# [12]: Shaffer et al. PRE 95 013206 (2017)
# [13]: Boercker and Pollock PRA 36 1779 (1987)
# [14]: https://lammps.sandia.gov/doc/fix_ave_correlate.html
# [15]: Scheiner et al. PRE 100, 043206 (2019)
# [16]: https://lammps.sandia.gov/doc/Build.html
#
#-------------------------------------------------------------------
#
# 2020.08.08 Created                                 TPZ
#            (copied and modified from BIM_D_PyL_1.0)
# 2020.08.28 Modified BIM_LAMMPS_D_Short             TPZ
#            (transform position arg into kwarg,
#            added mpi4py for PyLammps functioning 
#            normally, added proper way of generating
#            random seed numbers using mpi4py)
# 2020.09.07 Modified                                TPZ
#            (Renamed to LAMMPS_Diff.py, added OCP)
# 2020.09.19 Modified                                TPZ
#            (Commented BIM codes, directly 
#            generalized BIM for OCP, deleted OCP)
# 
#-------------------------------------------------------------------

import os
# supress multithreading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from lammps import PyLammps
import numpy as np
from random import randint
from ReadInput import simulation, species
from mpi4py import MPI

def OCP_BIM_D(sim, iTraj, neigh_one = 5000, neigh_page = 50000):
  """
  OCP_BIM_D(sim, neigh_one = 5000, neigh_page = 50000):
  Simulate one component fluid, or binary mixtures, which has either
  inverse polynomial or long-range Coulombic interactions, and calculate
  for one component fluid:
  the diagonal term of VACF, and its integral 
  for binary mixture:
  the diagonal term of VACF, and its integral for each species, and
  particle current autocorrelation function (PCACF).

  For each spatial dimension (i.e. for 3D, the x-, y- and z- 
  components), and the 3 cross terms (j_x(0)j_y(t), j_x(0)j_z(t), 
  j_y(0)j_z(t) in order) of PCACF for quantifying noise.
  PCACFs are calculated by fix ave/correlate command in LAMMPS [14]. 
  VACFs are calculated by assigning compute VACF command every given 
  number of timesteps. 
  For the meaning of parameters, please see the ReadInput.py and the 
  input file that the simulation class in ReadInput.py reads.
  The simulation length is controlled by the length (NLength), number
  of time origins (NReps) and the intervals between the time origins
  (NInt) of VACF. The total sampling step is NReps*NInt+NLength. 
  For convenience, this simulation is run under T* = 1 condition.
  The units used in here are in dimensionless LJ units. For units 
  and expressions of plasma-related variables and parameter units, 
  please see [6], which is based on [10], [11], [12] and [13].
  -------------------------------------------------------------------
  input arguments:

  sim: the simulation class file, which contains all the input 
  information of the simulation.

  iTraj: integer, the trajectory number, to make use of the analysis
  tools, please start the iTraj from 0.

  neigh_one: optional, integer, default = 5000, the maximum length of 
  neighbor list for one particle. Increase when LAMMPS complain 
  "neighbor list overflow, boost neigh_modify one". This usually happens
  when the density is large.

  neigh_page: optional, integer, default = 50000, the maximum pages of 
  neighbor list for one particle. LAMMPS official document suggests 
  setting neigh_page to be at least 10x of neigh_one, or otherwise 
  LAMMPS will complain.
  -------------------------------------------------------------------
  """
  
  # supress multithreading
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"

  # extract values from simulation class
  NReps, NLength, NInt, NFreq, NEvery = sim.NReps, sim.NLength, sim.NInt, sim.NFreq, sim.NEvery
  NEqm, tStep = sim.NEqm, sim.tStep
  EqmLogStem, ProdLogStem, FileStem = sim.EqmLogStem, sim.ProdLogStem, sim.FileStem
  NDim, NType, PairStyle, cutoff, AtomInfo = sim.NDim, sim.NSpecies ,sim.PairStyle, sim.cutoff, sim.species 
  NTotal = sim.NTotal
  # file stem name for Dii and D12
  DiiFileStem = FileStem +"_"+str(iTraj) +  "_Dii_"
  D12FileStem = FileStem +"_"+str(iTraj) + "_D12"

  if sim.PairStyle == "Coul":
    aWS, omega_p1, NGrids = sim.aWS, sim.omega_p1, sim.NGrids
  if sim.PairStyle == "poly":
    numden, polyCoeff = sim.numden, sim.polyCoeff
  
  # initiate PyLammps
  L = PyLammps()

  L.log(EqmLogStem+"_"+str(iTraj)+".txt") 

  L.variable("NEvery", "equal", NEvery)
  
  # random number generator for random number seeds by using lambda variable, 
  # so that two calls won't produce the same sequence of random numbers 
  RNG = lambda: randint(1, 100000)

  L.units("lj")
  L.dimension(NDim)
  L.boundary("p p p")

  if PairStyle == "Coul":
    L.atom_style("charge")
  else:
    L.atom_style("atomic")
  
  if PairStyle == "Coul":
  # Box volume (3D), box area (2D) and box length from Wigner-Seitz radius 
    if NDim == 2:
      BoxArea   =      np.pi * (aWS*aWS)     * NTotal
      BoxLength = BoxArea ** (1./2.)
    else:
      BoxVol    = 4. * np.pi * (aWS*aWS*aWS) * NTotal / 3.
      BoxLength = BoxVol  ** (1./3.)
  else:
    if NDim == 2:
      BoxArea   = NTotal / numden
      BoxLength = BoxArea ** (1./2.)
    else:
      BoxVol    = NTotal / numden
      BoxLength = BoxVol  ** (1./3.)

  # calculate timestep which is measured in LJ units
  if PairStyle == "Coul":
  	# Inverse plasma frequency of lighter species as a measure of time
    tp = 1. / omega_p1
    dt = tStep * tp 
  else:
    dt = tStep

  #-------------------------------------------------------------------
  # create box
  # create total region
  if NDim == 2:
    L.region("box block", 0., BoxLength, 0., BoxLength, 0., 1.) 
  if NDim == 3:
    L.region("box block", 0., BoxLength, 0., BoxLength, 0., BoxLength)

  #create simulation box
  L.create_box(NType, "box") 
  
  # create NType number of random numbers
  RandCreate = []
  # only ask the processor w/ rank 0 to generate random numbers
  if MPI.COMM_WORLD.rank == 0:
    for iType in range(NType):
      RandCreate.append(RNG())
  # broadcast (distribute) the generated random numbers to every processor
  RandCreate = MPI.COMM_WORLD.bcast(RandCreate, root=0)

  # create and set atoms 
  for iType in range(NType):
    L.create_atoms(iType+1, "random", int(AtomInfo[iType].number), RandCreate[iType] , "NULL")
  
  # Particle mass and charge
  for iType in range(NType):
    L.mass(iType+1, AtomInfo[iType].mass) 
    if PairStyle == "Coul":
      L.set("type", iType+1, "charge", AtomInfo[iType].charge)   
  
  # set the timestep
  L.timestep(dt) 
  
  # check neighbor parameters
  L.neigh_modify("delay", 0, "every", 1)
  
  # interaction style
  if PairStyle == "Coul":
    L.pair_style("coul/long", cutoff * aWS)
    L.pair_coeff("* *") 
    L.kspace_style("pppm", 1.0e-5)
    # NGrids need to be integer value of 2, 3 or 5
    L.kspace_modify("mesh", NGrids, NGrids, NGrids) 
    # Change tabinner to be smaller than default so that the real space
    # potential calculation can be benefitted by using the table; the 
    # default inner cutoff for using spline table is too large, larger
    # than the cutoff between direct/inverse cutoff of coul/long potential
    L.pair_modify("tabinner", 0.1 * aWS)
    # neighbor list need to be increased accordingly if the density (Gamma)
    # is too high, the "page" value need to be at least 10x the "one" value
    L.neigh_modify("one", neigh_one, "page", neigh_page)
  elif PairStyle == "poly":
    L.pair_style("poly")
    # Initialize
    polyPower = np.size(polyCoeff)
    polyPairCoeff = np.zeros((3, polyPower))
    # Three pairs of interactions, the coefficients are proportional to
    # the product of the charge of the pair. Pair 0 is Type (1,1)
    polyPairCoeff[0,:] = polyCoeff * AtomInfo[0].charge*AtomInfo[0].charge
    # pair 1 is Type (2,2), pair 2 is Type (1,2), only applicable to binary
    # mixtures
    if NType == 2:
      polyPairCoeff[1,:] = polyCoeff * AtomInfo[1].charge*AtomInfo[1].charge
      polyPairCoeff[2,:] = polyCoeff * AtomInfo[0].charge*AtomInfo[1].charge
    L.pair_coeff(1, 1, cutoff, polyPower, ''.join(str(iCoeff) + ' ' for iCoeff in polyPairCoeff[0,:]))
    # binary mixtures only
    # Both pair (1,2) and (2,1) need to be specified for pair coefficients
    if NType == 2:
      L.pair_coeff(2, 2, cutoff, polyPower, ''.join(str(iCoeff) + ' ' for iCoeff in polyPairCoeff[1,:]))
      L.pair_coeff(1, 2, cutoff, polyPower, ''.join(str(iCoeff) + ' ' for iCoeff in polyPairCoeff[2,:]))
      L.pair_coeff(2, 1, cutoff, polyPower, ''.join(str(iCoeff) + ' ' for iCoeff in polyPairCoeff[2,:]))
    # neighbor list need to be increased accordingly if the density 
    # is too high, the "page" value need to be at least 10x the "one" value
    L.neigh_modify("one", neigh_one, "page", neigh_page)
  else:
    raise Exception("the given potential style is not supported")

  #-------------------------------------------------------------------
  # Grouping based on types
  for iType in range(NType):
    L.group("Type"+str(iType+1), "type", iType+1)
  
  # generate a random number for setting velocity
  RandV = 0
  # only let the rank 0 processor to generate
  if MPI.COMM_WORLD.rank == 0:
    RandV = RNG()
  # broadcast to all the processors
  RandV = MPI.COMM_WORLD.bcast(RandV,root=0)
  L.velocity("all create 1.0", RandV)
  
  # Integrator set to be verlet
  L.run_style("verlet")
  # Nose-Hoover thermostat, the temperature damping parameter 
  # is suggested in [7]
  L.fix("Nose_Hoover all nvt temp", 1.0, 1.0, 100.0*dt) 

  #-------------------------------------------------------------------
  # Minimizing potential energy to prevent extremely high potential 
  # energy and makes the system reach equilibrium faster
  L.minimize("1.0e-6 1.0e-8 1000 10000")
  #-------------------------------------------------------------------
  # Equilibration run
  # log for equilibrium run
  
  L.reset_timestep(0)
  L.thermo(1000)
  # Equilibriation time
  L.run(NEqm)

  #-------------------------------------------------------------------
  # Production run
  # unfix NVT
  L.unfix("Nose_Hoover")
  # fix NVE, energy is conserved, not using NVT because T requires 
  # additional heat bath
  L.fix("NVEfix all nve") 
  L.reset_timestep(0)

  L.log(ProdLogStem+"_"+str(iTraj)+".txt") 

  # Labels in string texts for computes, fixes and variables that are 
  # labelled by x, y, and z
  DimLabel = ["x", "y", "z"]
  # particle current, for binary species only
  if NType == 2:
    # Define variables for mole fraction of two species 
    for iType in range(NType):
      L.variable("MolFrac"+"_"+str(iType+1), "equal", AtomInfo[iType].molfrac)
    # Define variables for particle currents 
    for iDim in range(3):
      L.variable("ParticleCurrent"+DimLabel[iDim], "equal",\
      "v_MolFrac_1*v_MolFrac_2*"+ \
      "(vcm(Type1,"+DimLabel[iDim]+")-"+ \
       "vcm(Type2,"+DimLabel[iDim]+"))") 
      
    # Set up fix ave/correlate for particle currents.
    L.fix("VACF"+"_"+"12", "all", "ave/correlate", NEvery, int(NLength/NEvery) + 1, NFreq,
    "v_ParticleCurrentx", "v_ParticleCurrenty", "v_ParticleCurrentz", 
    "type auto/lower", "ave running", "file", D12FileStem+".txt")

  # create two lists that stores the start and the end of each VACF calculations
  # the time origins are in the StartList, the ends are in the EndList
  # NReps is the number of time origins, NInt is the timesteps between consecutive 
  # origins, NLength is the timestep length of each VACF
  StartList = np.arange(0, NReps*NInt, NInt)
  EndList = np.arange(NLength, NLength+NReps*NInt, NInt)
  
  # How VACF and the integral are essentially computed:
  # Initiate compute VACF, store them by using `fix vector` and integrate by 
  # using `trap()` [3]
  # Then store the VACFs and the integral by `fix ave/time` command (see [8])
  # When the VACF calculation comes to the end, unfix/uncompute all the fixes and 
  # computes that are used for VACF calculations

  # initialize the VACF start and end counter 
  iStart = 0
  iEnd = 0
  while iEnd != NReps: 
    while iStart != NReps:
      # if the begin time of the next new VACF is earlier than the end time of the
      # oldest VACF that is running, start the new VACF
      if StartList[iStart] < EndList[iEnd]:
        for iType in range(NType):
          # define compute for VACF     
          L.compute("VACF"+"_"+str(iType+1)+"_"+str(iStart), "Type"+str(iType+1), "vacf")
          for iDim in range(1,4):
            # define fix vector for stacking VACFs
            L.fix(     "VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "all", "vector", NEvery,
            "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"["+str(iDim)+"]")
            # integrate with trapezium rule
            L.variable("Diff"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "equal",
            "${NEvery}*dt*trap("+"f_VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart)+")")
          # write out the VACFs by using fix ave/time
          L.fix("VACFout"+"_"+str(iType+1)+"_"+str(iStart), "all", "ave/time", 1, 1, NEvery, 
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[1]", "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[2]",
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[3]", 
          "v_Diffx"+"_"+str(iType+1)+"_"+str(iStart), "v_Diffy"+"_"+str(iType+1)+"_"+str(iStart), 
          "v_Diffz"+"_"+str(iType+1)+"_"+str(iStart), 
          "file", DiiFileStem+"Type"+str(iType+1)+"_"+str(iStart)+".txt", "mode scalar")
        # if this is the last VACF to be started, run the simulation until the next VACF end comes
        if iStart == (NReps - 1):
          L.run(EndList[iEnd]-StartList[iStart])
        # if not, run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        else:
          L.run(min(EndList[iEnd], StartList[iStart+1])-StartList[iStart])
        # increast the count of new VACF started by 1
        iStart += 1
      
      # if the begin time of the next new VACF is later than the end time of the
      # oldest VACF that is running, terminate the oldest VACF
      elif StartList[iStart] > EndList[iEnd]:
        # unfix and uncompute
        for iType in range(NType):
          for iDim in range(1,4):
            # unfix stacking VACF
            L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
          # unfix the output of VACF and the integral
          L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
          # uncompute for VACF
          L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))
        # run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        L.run(min(StartList[iStart], EndList[iEnd+1]) - EndList[iEnd])
        # increast the count of VACF terminated by 1
        iEnd += 1
      
      # if the begin time of the next new VACF and the end time of the oldest VACF that
      # is running happens at the same time, terminate the old and start the new
      else:
        for iType in range(NType):
          # fix and compute
          # define compute for VACF     
          L.compute("VACF"+"_"+str(iType+1)+"_"+str(iStart), "Type"+str(iType+1), "vacf")
          for iDim in range(1,4):
            # define fix vector for stacking new VACFs
            L.fix(     "VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "all", "vector", NEvery,
            "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"["+str(iDim)+"]")
            # integrate the new VACF with trapezium rule
            L.variable("Diff"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart), "equal",
            "${NEvery}*dt*trap("+"f_VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iStart)+")")
            # unfix stacking VACF
            L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
          
          # write out the new VACF and integral
          L.fix("VACFout"+"_"+str(iType+1)+"_"+str(iStart), "all", "ave/time", 1, 1, NEvery, 
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[1]", "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[2]",
          "c_VACF"+"_"+str(iType+1)+"_"+str(iStart)+"[3]", 
          "v_Diffx"+"_"+str(iType+1)+"_"+str(iStart), "v_Diffy"+"_"+str(iType+1)+"_"+str(iStart), 
          "v_Diffz"+"_"+str(iType+1)+"_"+str(iStart), 
          "file", DiiFileStem+"Type"+str(iType+1)+"_"+str(iStart)+".txt", "mode scalar")
          
          # unfix output of VACF
          L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
          # uncompute VACF
          L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))
        
        # if this is the last VACF to be started, run the simulation until the next VACF end comes
        if iStart == (NReps - 1):
          L.run(EndList[iEnd+1] - EndList[iEnd])
        # if not, run the simulation until either the next running VACF comes to the end or the next new
        # VACF need to be started
        else:
          L.run(min(EndList[iEnd+1], StartList[iStart+1]) - EndList[iEnd])
        # increase both counts by 1
        iStart += 1
        iEnd += 1
    
    # after all the VACFs are started, run the rest of the simulation and terminate those remaining runs
    # in order
    for iType in range(NType):
      for iDim in range(1,4):
        L.unfix("VACF"+DimLabel[iDim-1]+"_"+str(iType+1)+"_"+str(iEnd))
      L.unfix("VACFout"+"_"+str(iType+1)+"_"+str(iEnd))
      L.uncompute("VACF"+"_"+str(iType+1)+"_"+str(iEnd))

    if iEnd != (NReps - 1):
      L.run(EndList[iEnd+1] - EndList[iEnd])
    iEnd += 1


