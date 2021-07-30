#-------------------------------------------------------------------
#
#                     BIM_D_PyL_1.1/post_proc.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# This module contains several functions for post-processing of
# binary mixture (BIM) simulation data, with some presets that serves
# as examples to use these functions. The module contains the 
# following functions:
# - ReadVACFSingle: read single VACF and cumulate integral data
# - ReadPCACFSingleRep: read single PCACF data and evaluate integral
# - CumulateInt: evaluate Green-Kubo integral cumulatively
# - MeanSD: calculate mean and SD of a set of data
# - SEMCI: calculate SEM and CI of a set of data for given confidence 
# level
# - SelectSelfData: generate a list of tuples represent selected data 
# - SelectMutualData: generate a list of tuples represent selected data
# - SelfDiffSample: get the samples for self-diffusion coefficients 
# based on the list of tuples generated earlier
# - MutualDiffSample: get the samples for mutual-diffusion coefficients 
# based on the list of tuples generated earlier
# - ErrorBarPlot: generate a plot with error bars
# - SaveFile: save data with mean, SD, SEM, CI.
# and example preset functions:
# - SelfDiffAll: get all the self-diffusion coefficients, and analyse
# - MutualDIffAll: get all the mutual-diffusion coefficients and 
# analyse
# - PCACFAll: get all the (diagona) PCACF and analyse
# - ACFACF: get the averaged ACF of ACF
# - ChiSquare: do Chi square test for a range of ACF and test the 
# goodness of fit against a normal distribution with the same sigma
# and zero mean. Note that the sample data are dependently sampled
# from (assumed) identical distribution, so the test statistics 
# would be very high and p-value is nearly always zero.
# 
# Module dependencies:
# The versions below are the ones that I use when developing the 
# code. Earlier/later version may or may not work. 
#
# python: 3.8.5, with packages
# - numpy: 1.19 mainly for array manipulations
# - pandas: 1.1.0 for data intake
# - scipy: 1.5.2 for stats tools
# - matplotlib: 3.3.2 for pyplot
# and files
# - ReadInput.py: for extracting simulation inputs from the simulation
# class
#
#-------------------------------------------------------------------
#
# 2020.08.16 Created                                 TPZ
# 2020.08.19 Added comments                          TPZ
# 2020.08.24 Added comments, modified                TPZ
#            (Deleted "12" in PCACF file name) 
# 2020.09.02 Modified                                TPZ
#            (Added some statistical analysis in the
#            end)
# 2020.09.04 Modified                                TPZ
#            (Added MutualDiffSample, ReadPCACFSingle)
# 2020.09.28 Modified                                TPZ
#            (Added comments)
# 2021.04.17 Modified                                TPZ
#            (replace several "input.txt" by InputFile
#            replace the int() function for determining
#            iAvg by floor())
#-------------------------------------------------------------------

import os
# suppress python multithreading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from random import sample 
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import mlab as mlab
from ReadInput import simulation
from math import floor
import scipy.stats

def ReadVACFSingle(*, iTraj, iRep, iType, NLength, NDim, FileStem):
  """
  ReadVACFSingle read VACF and the cumulative integral from the output 
  texts with specified trajectory label (iTraj, start from 0), specified 
  repetition label within the trajectory (iRep, start from 0), and
  specified species (iType, 0 for Type 1, 1 for Type 2). The number of 
  dimensions of the simulation is specified by NDim, and the FileStem 
  is the base name of the data. The stored VACF and the cumulant data are 
  named as "FileStem_Dii_{iTraj}_{iType+1}_{iRep}.txt".
  ----------------------------------------------------------------------
  Input (all are keyword arguments, can be input in any order):
  
  iTraj: integer, the trajectory label that specifies which trajectory 
  to be read, taken in range between 0 and NTraj-1 (so total NTraj of 
  trajectories are available).    
  
  iRep: integer, the repetition label that specifies which repetition
  within a trajectory to be read, takein in range between 0 and NReps-1
  (so total NTraj of trajectories are available).
  
  iType: integer, specifies which species data to be read, 0 for Type 1, 
  1 for Type 2. Types are ordered ascendingly by their weight.
  
  NLength: integer, the total number of timesteps to be read.
  
  NDim: the number of dimensions of the simulation (more precisely, the
  number of dimensions that VACF and its cumulative integral have been 
  recorded in the data files). 3 if all the x-, y- and z-components are
  recorded, 2 if only any of the three are recorded.
  
  FileStem: string, the stem name of the reading file. The full name of 
  a file specified by iTraj, iRep and iType is:
  "FileStem_Dii_{iTraj}_{iType+1}_{iRep}.txt".
  ----------------------------------------------------------------------
  Output (in order):
  
  VACFSingle: a 2 dimensional numpy array with size ((NLength+1),3), it 
  stores the VACF for specified trajectory, repetition, dimensionality
  and species, for specified number of timestep. Its axis 0 is the number
  of timestep, and its axis 1 is the dimension, for 3D, 0 is for x, 1 is 
  for y, 2 is for z.
  
  SelfDiffSingle: a 2 dimensional numpy array with size ((NLength+1),3), it 
  stores the self-diffusion coefficient that is evaluated by the cumulative 
  integral (done by LAMMPS) for specified trajectory, repetition, 
  dimensionality and species, for specified number of timestep. Its axis 0 
  is the number of timestep, and its axis 1 is the dimension, for 3D, 0 is 
  for x, 1 is for y, 2 is for z.
  """

  # Initialize
  VACFSingle     = np.zeros((NLength+1,NDim))
  SelfDiffSingle = np.zeros((NLength+1,NDim))

  # Read csv by pandas module and then convert to numpy array. Pandas read_csv()
  # is faster than numpy loadtxt(), but requires more memory (some discussions
  # can be found in stackoverflow. 
  DiffVACF = pd.read_csv(
  # file name
  FileStem + "_" + str(iTraj) + "_Dii_Type" + str(iType+1) + "_" + str(iRep) + ".txt",
  skiprows = list(range(2)), nrows = (NLength+1), usecols = list(range(1, NDim*2+1)),
  delimiter=' ', header=None
  ).to_numpy() 
  # extract from DiffVACF
  VACFSingle     = DiffVACF[:, 0   :  NDim]
  SelfDiffSingle = DiffVACF[:, NDim:2*NDim]
  return VACFSingle, SelfDiffSingle

def ReadPCACFSingleRep(*, iTraj, iAvg, NLength, NEvery = 1, dt, NDim, FileStem):
  """
  ReadPCACFSingleRep read the PCACFs recorded in trajectory labelled by 
  iTraj (in range from 0 to NTraj-1), at iAvg repetition of the PCACF
  average calculations (there are int(NLength+NInt*NReps/NFreq) in total).
  It then performs the Green-Kubo integral cumulatively for each dimensional
  elements (including all the diagonal terms, such as xx, yy, zz, in 3D,
  and off diagonal terms, such as  xy, xz, yz in 3D), which is the binary
  diffusion coefficients for diagonal components and expect to be 0 for 
  the off-diagonal components. The number of dimensions of the simulation 
  is specified by NDim, and the FileStem is the base name of the data. 
  The stored PCACF data are named as "FileStem_D12_{iTraj}.txt".
  ----------------------------------------------------------------------
  Input (all are keyword arguments, can be input in any order):
  
  iTraj: integer, the label for trajectory, range from 0 to NTraj-1.
  
  iAvg: integer, the label for the reported PCACF averages, count from 0,
  there are int(NLength+NInt*NReps/NFreq) of them in total.
  
  NLength: integer, the total number of timesteps to be read. This 
  NLength does not necessary need to be the same as the NLength in 
  ReadVACFSingle(), as it is actually the NFreq in LAMMPS ave/correlate,
  which is calculated based on int(tFreq/tStep) of the simulation input
  file. For now NLength can only be the same as int(tFreq/tStep) as 
  specified in the simulation input.
  
  NEvery: integer, optional, default = 1, the sampling frequency for 
  calculating PCACF in the simulation. In a MD simulation, fluxes are 
  calculated every NEvery steps. This quantity is needed because the 
  interval between each step is needed for performing integration over
  time. 
  
  dt: integer, the timestep length of the simulation. It can be written
  in any unit, as long as you track the unit for the calculated mutual-
  diffusion coefficient accordingly (e.g. if you are using omega_p^-1 
  instead of tau, then the calculated D12 carries omega_p^-1 as well).
  This quantity is needed because the interval between each step is 
  needed for performing integration over time. 
  
  NDim: the number of dimensions of the simulation (more precisely, the
  number of dimensions that PCACF and its cumulative integral have been 
  recorded in the data files). 3 if all the x-, y- and z-components and
  their off-diagonal terms are recorded, 2 if only any of the three and
  their off-diagonal terms are recorded.
  
  FileStem: string, the stem name of the reading file. The full name of 
  a file specified by iTraj is:
  "FileStem_D12_{iTraj}.txt".
  ----------------------------------------------------------------------
  Output (in order):
  
  PCACF: a 3 dimensional numpy array with size 
  (NReps, NLength+1, NDim*(NDim+1)/2), it stores all the PCACF for
  the specified trajectory, all repetition and all dimensions for specified  
  number of timestep. Its axis 0 is the number of trajectory, 1 is the 
  number of repetition, 2 is the number of timestep, 3 is the dimensions. 
  For 2D, the dimensions are (from 0): xx, yx, yy (if the simulation is 
  in xy plane), for 3D, the dimensions are (from 0): xx, yx, yy, zx, zy,
  zz.
  
  Diff12: a 3 dimensional numpy array with size 
  (NReps, NLength+1, NDim*(NDim+1)/2), it stores all the mutual-
  diffusion coefficients for the specified trajectory, all repetition  
  and all dimensions for specified number of timestep. Its axis 0 is the 
  number of trajectory, 1 is the number of repetition, 2 is the number 
  of timestep, 3 is the dimensions. For 2D, the dimensions are (from 0):
  xx, yx, yy (if the simulation is in xy plane), for 3D, the dimensions 
  are (from 0): xx, yx, yy, zx, zy, zz.
  """
  
  # Initialize,
  PCACF     = np.zeros((NLength+1, int(NDim*(NDim+1)/2)))
  Diff12    = np.zeros((NLength+1, int(NDim*(NDim+1)/2)))

  # Read csv by pandas module and then convert to numpy array, PCACFTemp is
  # a temporary numpy array that stores all the data from the file. 
  PCACF = pd.read_csv(
  # file name
  FileStem + "_" + str(iTraj) + "_D12" + ".txt",
  # skip header rows and the rows that label which divides the reps
  skiprows = list(range(NLength+5+1+(iAvg)*(NLength+2))), 
  nrows = (NLength+1), usecols = [3,4,5,6,7,8], delimiter = ' ', header=None
  ).to_numpy()

  for iDim in range(int(NDim*(NDim+1)/2)):
    # integrate
    Diff12[:, iDim] = CumulateInt(PCACF[:, iDim], dt*float(NEvery)) 
  return PCACF, Diff12

def CumulateInt(InputArray, dx):
  """
  CumulateInt calculate the running integral of a given vector (y), with 
  given fixed dx between each consecutive y value. The integral of y is 
  evaluated using trapezium rule.
  ----------------------------------------------------------------------
  Input (in order):
  
  InputArray: a 1 dimensional vector (a numpy vector or a list with floats).
  The data to be integrated.
  
  dx: float, the fixed interval between each step (interval) in InputArray.
  ----------------------------------------------------------------------
  Output:
  
  Integral: a 1 dimensional numpy vector with the same length as InputArray.
  It stores the cumulative integral.
  """
  Integral = np.zeros(InputArray.size)
  Integral[0] = 0.
  for iStep in range(InputArray.size-1):
    Integral[iStep+1] = Integral[iStep] + (InputArray[iStep] + InputArray[iStep+1])
  Integral = Integral*dx*0.5
  return Integral

def MeanSD(InputSample):
  """
  MeanSD determine the mean and unbiased estimate of standard deviation of 
  the population. The averages are calculated along the axis 0 (the first 
  index), so that it can handle ACF and transport coefficient data which
  is in form of a 2d numpy array of the size (NSample,(NLength/NEvery)+1)
  ----------------------------------------------------------------------
  Input:
  
  InputSample: a 2 dimensional numpy array with size (NSample,*), where *
  denotes an arbitrary length.
  ----------------------------------------------------------------------
  Output (in order):
  
  OutputMean: a 1 dimensional numpy vector with the same size as the axis
  1 of InputSample (denoted as *). It is the series of the main of the
  input sample along the axis 0.
  
  OutputSD: a 1 dimensional numpy vector with the same size as the axis
  1 of InputSample (denoted as *). It is the unbiased estimate of standard
  deviation of the population based on the input sample data.
  """
  OutputSD = np.std(InputSample, axis=0, ddof=1)
  OutputMean = np.mean(InputSample, axis=0)
  return OutputMean, OutputSD

def SEMCI(InputSample, confidence):
  """
  SEMCI determine the standard error of mean and confidence interval along 
  the axis 0 for a given set of input data.  
  ----------------------------------------------------------------------
  Input (in order):
  
  InputSample: a 2 dimensional numpy array with size (NSample,*), where *
  denotes an arbitrary length.
  
  confidence: float, > 0 and < 1, the level of the confidence interval, 
  e.g. 0.95, 0.99.
  ----------------------------------------------------------------------
  Output (in order):
  
  SEM: a 1 dimensional numpy vector with the same size as the axis
  1 of InputSample (denoted as *). It is the standard error of the mean
  based on the population.
  
  CI: a 1 dimensional numpy vector with the same size as the axis
  1 of InputSample (denoted as *). It is the confidence interval of the 
  mean based on the population and the given value for confidence level.
  """
  NSample = InputSample.shape[0]
  SEM = scipy.stats.sem(InputSample, axis=0)
  # calculate CI based on the t distribution
  CI = SEM * scipy.stats.t.ppf((1 + confidence) / 2., NSample-1)
  return SEM, CI

def SelectSelfData(*, mode, NSample, NTraj, NReps, NDim):
  """
  SelectSelfData generates a list of random but non-repetitive tuples 
  which specifies the sample data to be picked up for statistical 
  analysis for self-diffusion coefficient and VACF. The tuple may 
  contains labels of trajectories, repetitions and dimensions. They 
  depend on the mode of data selection. See details in output 
  description.
  ----------------------------------------------------------------------
  Input (all are keyword arguments, can be input in any order):
  
  mode: string, can only be one of the followings:
  "All", "OneTraj", "OneRep", "x", "y", "z"
  
  NSample: integer, the number of samples to be picked up
  
  NTraj: integer, the total number of trajectories to be chosen from
  
  NReps: integer, the total number of repetitions to be chosen from
  
  NDim: integer, the number of simulation dimensions
  ----------------------------------------------------------------------
  Output: 
  
  RandDataIndex: a list with NSample elements, each item is a tuple,
  which depends on the mode of operation:
  -"All": the sample data are picked up from all the NTraj trajectories,
  NReps repetitions and all the dimensions. The tuples are in the form 
  (iTraj, iRep, iDim), where iTraj, iRep, iDim are the labels for a
  specific trajectory, repetition and dimension.
  -"x", "y" and "z": the sample data are picked up from all the NTraj 
  trajectories, NReps repetitions but only one dimension. The tuples 
  are in the form (iTraj, iRep), where iTraj, iRep are the labels for a
  specific trajectory and repetition.
  -"OneTraj": the sample data are picked up from one trajectory
  all the NReps repetitions and all the dimensions. The tuples are in  
  the form (iRep, iDim), where iRep, iDim are the labels for a
  specific repetition and a dimension.
  -"OneRep": the sample data are picked up from all the trajectories
  at the same repetition and all the dimensions. The tuples are in  
  the form (iTraj, iDim), where iTraj, iDim are the labels for a
  specific trajectory and a dimension.
  """
  if mode == "All":
    RandDataIndex = sample(list(product(range(NTraj), range(NReps), range(NDim))), k=NSample)
    return RandDataIndex
  elif mode in {"x", "y", "z"}:
    RandDataIndex = sample(list(product(range(NTraj), range(NReps))), k=NSample)
    return RandDataIndex
  elif mode == "OneTraj":
    RandDataIndex = sample(list(product(range(NReps), range(NDim))), k=NSample)
    return RandDataIndex
  elif mode == "OneRep":
    RandDataIndex = sample(list(product(range(NTraj), range(NDim))), k=NSample)
    return RandDataIndex
  else:
    raise Exception("mode need to be All or x or y or z or OneTraj or OneRep")
  
def SelectMutualData(*, mode, NSample, NTraj, NDim):
  """
  SelectMutualData generates a list of random but non-repetitive tuples 
  which specifies the sample data to be picked up for statistical 
  analysis for PCACF and mutual-diffusion coefficient. The tuple may 
  contains labels of trajectories, repetitions and dimensions. They 
  depend on the mode of data selection. See details in output 
  description.
  ----------------------------------------------------------------------
  Input (all are keyword arguments, can be input in any order):

  mode: string, can only be one of the followings:
  "Diag", "Cross", "x", "y", "z", "yx", "yz", "xz"

  NSample: integer, the number of samples to be picked up
  
  NTraj: integer, the total number of trajectories to be chosen from
  
  NDim: integer, the number of simulation dimensions
  ----------------------------------------------------------------------
  Output: 
  
  RandDataIndex: a list with NSample elements, each item is a tuple,
  which depends on the mode of operation:
  -"Diag": the sample data are picked up from all the NTraj trajectories
  and the dimensions of the diagonal terms e.g. xx, yy, zz. The tuples are 
  in the form (iTraj, iDim), where iTraj and iDim are the labels for a
  specific trajectory and dimension.
  -"Cross": the sample data are picked up from all the NTraj trajectories
  and the dimensions of the cross terms e.g. xy, yz, xz. For 3D simulation
  data, the tuples are in the form (iTraj, iDim), where iTraj and iDim are 
  the labels for a specific trajectory and dimension, for 2D simulation data,
  since the only cross term is xy (for simulations carried out in xy plane)
  the iDim is omitted.
  -"x", "y", "z", "yx", "yz", "xz": the sample data are picked up from
  all the NTraj trajectories but only one dimension. The tuples 
  are in the form (iTraj), where iTraj is the labels for a specific
  trajectory.
  """
  if NDim == 3:
    if mode == "Diag":
      RandDataIndex = sample(list(product(range(NTraj), [0,2,5])), k=NSample)
      return RandDataIndex
    elif mode == "Cross":
      RandDataIndex = sample(list(product(range(NTraj), [1,3,4])), k=NSample)
      return RandDataIndex
    elif mode in {"x", "y", "z", "yx", "zy", "zx"}:
      RandDataIndex = sample(list(product(range(NTraj))), k=NSample)
      return RandDataIndex
    else:
      raise Exception("mode need to be Diag or Cross or x or y or z or yx or zy or zx")
  elif NDim == 2:
    if mode == "Diag":
      RandDataIndex = sample(list(product(range(NTraj), [0,2])), k=NSample)
      return RandDataIndex
    elif mode in {"x", "y", "z", "yx", "zy", "zx", "Cross"}:
      RandDataIndex = sample(list(product(range(NTraj))), k=NSample)
      return RandDataIndex
    else:
      raise Exception("mode need to be Diag or Cross or x or y or z or yx or zy or zx")
  else:
    raise Exception("NDim must be 2 or 3")

def SelfDiffSample(*, RandDataIndex, mode, iType, iTraj = None, NTraj, iRep = None, NReps, NLength, NDim, FileStem):
  """
  SelfDiffSample formulate a set of sample VACF and self-diffusion 
  coefficient data, chosen by function SelectSelfData. 
  ----------------------------------------------------------------------
  Input (all are keyword arguments, can be input in any order):

  RandDataIndex: a list with NSample elements, each item is a tuple,
  which depends on the mode of operation:
  -"All": the sample data are picked up from all the NTraj trajectories,
  NReps repetitions and all the dimensions. The tuples are in the form 
  (iTraj, iRep, iDim), where iTraj, iRep, iDim are the labels for a
  specific trajectory, repetition and dimension.
  -"x", "y" and "z": the sample data are picked up from all the NTraj 
  trajectories, NReps repetitions but only one dimension. The tuples 
  are in the form (iTraj, iRep), where iTraj, iRep are the labels for a
  specific trajectory and repetition.
  -"OneTraj": the sample data are picked up from one trajectory
  all the NReps repetitions and all the dimensions. The tuples are in  
  the form (iRep, iDim), where iRep, iDim are the labels for a
  specific repetition and a dimension.
  -"OneRep": the sample data are picked up from all the trajectories
  at the same repetition and all the dimensions. The tuples are in  
  the form (iTraj, iDim), where iTraj, iDim are the labels for a
  specific trajectory and a dimension.

  mode: string, can only be one of the followings:
  "All", "OneTraj", "OneRep", "x", "y", "z"

  iType: integer, the species type label, 0 for Type 1, 1 for Type 2

  iTraj: integer, only needed when mode = "OneTraj", the trajectory
  label that specifies which trajectory to be read. It can be chosen
  in between 0 and NTraj-1.  

  NTraj: integer, the total number of trajectories to be chosen from

  iRep: integer, only needed when mode = "OneRep", the repetition
  label that specifies which repetition to be read. It can be chosen
  in between 0 and NReps-1. 

  NReps: integer, the total number of repetitions to be chosen from
  
  NLength: integer, the total number of timesteps to be read.
  
  NDim: integer, the number of simulation dimensions
  
  FileStem: string, the stem name of the reading file. The full name of 
  a file specified by iTraj, iRep and iType is:
  "FileStem_Dii_{iTraj}_{iType+1}_{iRep}.txt".
  ----------------------------------------------------------------------
  Output (in order):

  SampleVACFData: a 2 dimensional numpy array with size (NSample, NLength+1).
  It stores all the VACF data that are picked up.

  SampleDiiData: a 2 dimensional numpy array with size (NSample, NLength+1).
  It stores all the self-diffusion coefficient data that are picked up.
  """
  SampleDiiData = []
  SampleVACFData = []
  if mode == "OneTraj":
    pass
  elif mode == "All":
    # Initialize
    SelfDiff = np.zeros((NTraj, NReps, NLength+1, NDim))
    VACF     = np.zeros((NTraj, NReps, NLength+1, NDim))
    # Get a list of trajectory to be read, by using dictionary
    TrajList = list(dict.fromkeys([SampleTuple[:-1] for SampleTuple in RandDataIndex]))
    # Read necessary trajectory
    for iTuple in TrajList:
      VACF[iTuple[0], iTuple[1], :, :], SelfDiff[iTuple[0], iTuple[1], :, :] = ReadVACFSingle(
      # ReadVACFSingle(*, iTraj, iRep, iType, NLength, NDim, FileStem)
      iTraj = iTuple[0], iRep = iTuple[1], iType = iType, NLength = NLength, NDim = NDim, FileStem = FileStem
      )
    # Extract VACF and self-diffusion coefficients
    for iSample in RandDataIndex:
      SampleVACFData.append(np.transpose(VACF    [iSample[0], iSample[1], :, iSample[2]]))
      SampleDiiData.append (np.transpose(SelfDiff[iSample[0], iSample[1], :, iSample[2]]))
  elif mode in {"x", "y", "z"}:
    # Initialize
    SelfDiff = np.zeros((NTraj, NReps, NLength+1, NDim))
    VACF     = np.zeros((NTraj, NReps, NLength+1, NDim))
    # Get a list of trajectory to be read, by using dictionary
    TrajList = list(dict.fromkeys([SampleTuple[:-1] for SampleTuple in RandDataIndex]))
    # Read necessary trajectory
    for iTuple in TrajList:
      VACF[iTuple[0], iTuple[1], :, :], SelfDiff[iTuple[0], iTuple[1], :, :] = ReadVACFSingle(
      # ReadVACFSingle(*, iTraj, iRep, iType, NLength, NDim, FileStem)
      iTraj = iTuple[0], iRep = iTuple[1], iType = iType, NLength = NLength, NDim = NDim, FileStem = FileStem
      )
    modedict = {"x":0, "y":1, "z":2}
    # Extract VACF and self-diffusion coefficients
    for iSample in RandDataIndex:
      SampleVACFData.append(np.transpose(VACF    [iSample[0], iSample[1], :, modedict[mode]]))
      SampleDiiData.append (np.transpose(SelfDiff[iSample[0], iSample[1], :, modedict[mode]]))
  elif mode == "OneRep":
    # Initialize
    SelfDiff = np.zeros((NTraj, NLength+1, NDim))
    VACF     = np.zeros((NTraj, NLength+1, NDim))
    # Get a list of trajectory to be read, by using dictionary
    TrajList = list(dict.fromkeys([SampleTuple[:-1] for SampleTuple in RandDataIndex]))
    # Read necessary trajectory
    for iTuple in TrajList:
      VACF[iTuple[0], :, :], SelfDiff[iTuple[0], :, :] = ReadVACFSingle(
      # ReadVACFSingle(*, iTraj, iRep, iType, NLength, NDim, FileStem)
      iTraj = iTuple[0], iRep = iRep, iType = iType, NLength = NLength, NDim = NDim, FileStem = FileStem
      )
    for iSample in RandDataIndex:
      SampleVACFData.append(np.transpose(VACF    [iSample[0], :, iSample[1]]))
      SampleDiiData.append(np.transpose(SelfDiff[iSample[0], :, iSample[1]]))
  else:
    raise Exception("mode need to be All or x or y or z or OneTraj or OneRep")
  # convert the result data to numpy arrays
  SampleVACFData = np.array(SampleVACFData)
  SampleDiiData = np.array(SampleDiiData)
  return SampleVACFData, SampleDiiData

def MutualDiffSample(*, RandDataIndex, mode, NTraj, iAvg, NLength, NEvery, dt, NDim, FileStem):
  # Initialize
  SampleD12Data = []
  SamplePCACFData = []
  PCACF  = np.zeros((NTraj, NLength+1, int(NDim*(NDim+1)/2)))
  Diff12 = np.zeros((NTraj, NLength+1, int(NDim*(NDim+1)/2)))
  if mode in {"Cross", "Diag"}:
    # Get a list of trajectory to be read, by using dictionary
    TrajList = list(dict.fromkeys([SampleTuple[:-1] for SampleTuple in RandDataIndex]))
    # Read necessary trajectory
    for iTuple in TrajList:
      PCACF[iTuple[0], :, :], Diff12[iTuple[0], :, :] = ReadPCACFSingleRep(
      # ReadPCACFSingleRep(*, iTraj, iAvg, NLength, NEvery = 1, dt, NDim, FileStem)
      iTraj = iTuple[0], iAvg = iAvg, NLength = NLength, NEvery = NEvery, dt=dt, NDim = NDim, FileStem = FileStem
      )
    # Extract VACF and self-diffusion coefficients
    for iSample in RandDataIndex:
      SamplePCACFData.append(np.transpose(PCACF [iSample[0], :, iSample[1]]))
      SampleD12Data  .append(np.transpose(Diff12[iSample[0], :, iSample[1]]))
  elif mode in ("x", "y", "z", "yx", "zy", "zx"):
    # Get a list of trajectory to be read, by using dictionary
    TrajList = list(dict.fromkeys([SampleTuple[:-1] for SampleTuple in RandDataIndex]))
    # Read necessary trajectory
    for iTuple in TrajList:
      PCACF[iTuple[0], :, :], Diff12[iTuple[0], :, :] = ReadPCACFSingleRep(
      # ReadPCACFSingle(*, iTraj, NReps, NLength, NEvery = 1, dt, NDim, FileStem)
      iTraj = iTuple[0], iAvg = iAvg, NLength = NLength, NEvery = NEvery, dt=dt, NDim = NDim, FileStem = FileStem
      )
    if NDim == 2:
      modedict = {"x": 0, "yx":1, "y":2}
      if mode in {"z", "zx", "zy"}:
        raise Exception("for 2D simulation, mode can be only x or yx or y")
    if NDim == 3:
      modedict = {"x": 0, "yx":1, "y":2, "z":5, "zx":3, "zy":4}
    # Extract PCACF and mutual-diffusion coefficients
    for iSample in RandDataIndex:
      SamplePCACFData.append(np.transpose(PCACF [iSample[0], :, modedict[mode]]))
      SampleD12Data  .append(np.transpose(Diff12[iSample[0], :, modedict[mode]]))
  else:
    raise Exception("mode need to be Diag or Cross or x or y or z or yx or zy or zx")
  # convert the result data to numpy arrays
  SamplePCACFData = np.array(SamplePCACFData)
  SampleD12Data = np.array(SampleD12Data)
  return SamplePCACFData, SampleD12Data

def ErrorBarPlot(*,NLength, NEvery, dt, yData, yErr, xLabel = None, yLabel = None, FileName):
  """
  ErrorBarPlot generates plots with specified error bars.
  ----------------------------------------------------------------------
  NLength: integer, ACF recording step length
  NEvery: integer, ACF sampling step
  dt: float, length of time step, depends on the simulation time units
  yData: array with dimension 1 and length NLength, the selected y value
  yErr: array with dimension 1 and length NLength, the selected y error
  xLabel: string, the label for x axis
  yLabel: string, the label for y axis
  FileName: the output file name
  ----------------------------------------------------------------------
  Output:
  figure named FileName.png
  """
  plt.rc('font',family='serif',size=16)
  # plt.rc('text',usetex=True)
  plt.rc('xtick',labelsize='small')
  plt.rc('ytick',labelsize='small')
  fig = plt.figure(figsize=(7,5.25))
  ax = fig.add_subplot(1,1,1)
  # ax.legend(frameon=False)
  ax.errorbar(np.linspace(0.0, dt*NEvery*NLength, num = NLength+1), yData, yerr = yErr, color = "k", ecolor = "b")
  ax.set_xlabel(xLabel)
  ax.set_ylabel(yLabel)
  fig.savefig(FileName+".png")

def SaveFile(*, NSample, NEvery, dt, confidence, Mean, SD, SEM, CI, FileName):
  """
  SaveFile generates files that records the following informations in 
  order: time, mean, standard deviation, standard error of mean, 
  confidence interval at a given confidence level. Also in the header
  file, the number of samples and confidence level are saved.
  ----------------------------------------------------------------------
  NSample: integer, number of sample chosen
  NEvery: integer, ACF sampling step
  dt: float, length of time step, depends on the simulation time units
  Confidence: the confidence level for calculating confidence interval
  Mean, SD, SEM, CI: the mean, SD, SEM and CI of data
  FileName: the output file name
  ----------------------------------------------------------------------
  Output:
  text file named FileName.txt
  """
  np.savetxt(FileName+".txt", np.transpose([np.linspace(0.0, dt*NEvery*(Mean.size-1), num = Mean.size), Mean, SD, SEM, CI]), 
  header= " NSample = " + str(NSample) + " confidence = " + str(confidence) + "\n" + "Time Mean SD SEM CI",
  fmt = '%.6e')

#-------------------------------------------------------------------
# presets for analysis, these presets are served as examples

def SelfDiffAll(InputFile, NTraj, iSpecies):
  """
  average over all the self-diffusion data and get the 95% CI
  input (in order):
  InputFile: string, the name for simulation input file
  NTraj: integer, the number of trajectory
  iSpecies: integer, 0 for Type1, 1 for Type2

  output: plot and text file 
  """

  sim = simulation(InputFile)
  mode = "All"
  NReps = sim.NReps
  NDim = sim.NDim
  tStep = sim.tStep
  if sim.PairStyle == "Coul":
    dt = sim.tStep/sim.omega_p1
  else:
    dt = sim.tStep
  iRep = NReps-1
  
  NLength = sim.NLength
  NEvery = sim.NEvery
  FileStem = sim.FileStem
  
  NSample = int(NTraj*NDim*NReps)
  FileName = FileStem +"_Dii_Type"+str(iSpecies+1)+"_All"+"_NSample_"+str(NSample)
  
  RandDataIndex = SelectSelfData(mode = mode , NSample = NSample, NTraj = NTraj, NReps = NReps, NDim = NDim)
  SampleVACFData, SampleDiiData = SelfDiffSample(RandDataIndex = RandDataIndex, mode = mode, iType = iSpecies, NTraj = NTraj, 
  NReps = NReps, NLength = NLength, NDim = NDim, FileStem = FileStem)
  
  Mean, SD = MeanSD(SampleDiiData)
  Mean = Mean/(sim.aWS*sim.aWS*sim.omega_p)
  SD = SD/(sim.aWS*sim.aWS*sim.omega_p)
  SEM, CI = SEMCI(SampleDiiData, 0.95)
  SEM = SEM/(sim.aWS*sim.aWS*sim.omega_p)
  CI = CI/(sim.aWS*sim.aWS*sim.omega_p)
  
  if sim.PairStyle == "Coul":
    xLabel = r"$t \omega_{p1}$"
    yLabel = r"$D_{12}/a^2 \omega_p$"
  else:
    xLabel = r"$t/\tau$"
    yLabel = r"$D_{12}/\sigma^2\tau^{-1}$"
  
  ErrorBarPlot(NLength = NLength, NEvery = NEvery, dt = tStep, yData = Mean, yErr = CI, FileName = FileName, xLabel = xLabel, yLabel = yLabel)
  SaveFile(NSample = NSample, NEvery = NEvery, dt = tStep, confidence = 0.95, Mean = Mean, SD = SD, SEM = SEM, CI = CI, FileName = FileName)

def MutualDiffAll(InputFile, NTraj):
  """
  average over all the mutual-diffusion data and get the 95% CI
  input (in order):
  InputFile: string, the name for simulation input file
  NTraj: integer, the number of trajectory

  output: plot and text file 
  """

  sim = simulation(InputFile)
  mode = "Diag"
  NReps = sim.NReps
  NDim = sim.NDim
  tStep = sim.tStep
  if sim.PairStyle == "Coul":
    dt = sim.tStep/sim.omega_p1
  else:
    dt = sim.tStep
  iAvg = floor((NReps*sim.tInt+sim.tLength)/sim.tFreq)-1
  
  NLength = sim.NLength
  NEvery = sim.NEvery
  FileStem = sim.FileStem
  
  NSample = int(NTraj*NDim)
  FileName = FileStem +"_D12"+"_All"+"_NSample_"+str(NSample)
  
  RandDataIndex = SelectMutualData(mode = mode, NSample = NSample, NTraj = NTraj, NDim = NDim)
  SamplePCACFData, SampleD12Data = MutualDiffSample(RandDataIndex = RandDataIndex, mode = mode,
  NTraj = NTraj, iAvg = iAvg, NLength = NLength, NEvery = NEvery, NDim = NDim, dt = dt, FileStem = FileStem)
  
  Mean, SD = MeanSD(SampleD12Data) 
  Mean = Mean * (sim.NTotal/(sim.aWS*sim.aWS*sim.omega_p*sim.species[0].molfrac*sim.species[1].molfrac))
  SD = SD * (sim.NTotal/(sim.aWS*sim.aWS*sim.omega_p*sim.species[0].molfrac*sim.species[1].molfrac))
  SEM, CI = SEMCI(SampleD12Data, 0.95) 
  SEM = SEM * (sim.NTotal/(sim.aWS*sim.aWS*sim.omega_p*sim.species[0].molfrac*sim.species[1].molfrac))
  CI = CI * (sim.NTotal/(sim.aWS*sim.aWS*sim.omega_p*sim.species[0].molfrac*sim.species[1].molfrac))
  
  if sim.PairStyle == "Coul":
    xLabel = r"$t \omega_{p1}$"
    yLabel = r"$D_{12}/a^2 \omega_p$"
  else:
    xLabel = r"$t/\tau$"
    yLabel = r"$D_{12}/\sigma^2\tau^{-1}$"
  
  ErrorBarPlot(NLength = NLength, NEvery = NEvery, dt = tStep, yData = Mean, yErr = CI, FileName = FileName, xLabel = xLabel, yLabel = yLabel)
  SaveFile(NSample = NSample, NEvery = NEvery, dt = tStep, confidence = 0.95, Mean = Mean, SD = SD, SEM = SEM, CI = CI, FileName = FileName)

def PCACFAll(InputFile, NTraj):
  """
  average over all the PCACF data and get the 95% CI
  input (in order):
  InputFile: string, the name for simulation input file
  NTraj: integer, the number of trajectory

  output: plot and text file 
  """

  sim = simulation(InputFile)
  mode = "Diag"
  NReps = sim.NReps
  NDim = sim.NDim
  tStep = sim.tStep
  if sim.PairStyle == "Coul":
    dt = sim.tStep/sim.omega_p1
  else:
    dt = sim.tStep
  iAvg = floor((NReps*sim.tInt+sim.tLength)/sim.tFreq)-1
  
  NLength = sim.NLength
  NEvery = sim.NEvery
  FileStem = sim.FileStem
  
  NSample = int(NTraj*NDim)
  FileName = FileStem +"_PCACF"+"_All"+"_NSample_"+str(NSample)
  
  RandDataIndex = SelectMutualData(mode = mode, NSample = NSample, NTraj = NTraj, NDim = NDim)
  SamplePCACFData, SampleD12Data = MutualDiffSample(RandDataIndex = RandDataIndex, mode = mode,
  NTraj = NTraj, iAvg = iAvg, NLength = NLength, NEvery = NEvery, NDim = NDim, dt = dt, FileStem = FileStem)
  
  Mean, SD = MeanSD(SamplePCACFData) 
  SEM, CI = SEMCI(SampleD12Data, 0.95) 

  if sim.PairStyle == "Coul":
    xLabel = r"$t \omega_{p1}$"
  else:
    xLabel = r"$t/\tau$"
  yLabel = r"$PCACF$"
  
  ErrorBarPlot(NLength = NLength, NEvery = NEvery, dt = tStep, yData = Mean, yErr = CI, FileName = FileName, xLabel = xLabel, yLabel = yLabel)
  SaveFile(NSample = NSample, NEvery = NEvery, dt = tStep, confidence = 0.95, Mean = Mean, SD = SD, SEM = SEM, CI = CI, FileName = FileName)

def ACFACF(InputFile, InputACF, tStart, tEnd, ACFtLength):
  """
  determine ACF of ACF sampling start from tStart to tEnd, for 
  ACF length ACFtLength
  input (in order):
  InputFile: string, the name for simulation input file
  InputACF: the input ACF file
  tStart: the initial time for the sample data chosen
  tEnd: the final time for the sample data chose
  ACFtLength: the desired length of ACF in terms of t. 
  """
  sim = simulation(InputFile)
  NStart = int(tStart/sim.tStep)
  NEnd   = int(tEnd/sim.tStep)
  ACFNLength = ACFtLength / sim.tStep
  
  ACFData = pd.read_csv(InputACF,
  # skip header rows and the rows that label which divides the reps
  skiprows = list(range(NStart+2)),
  nrows = NEnd - NStart + 1, usecols = [1], delimiter = ' ', header=None
  ).to_numpy()
  
  ACFData = np.squeeze(ACFData)

  # ACF of ACF
  ACF2 = []
  for iTime in range(int((NEnd - NStart) / ACFNLength)):
    ACF2.append(np.correlate(
    ACFData[int(iTime*ACFNLength):int((iTime+1)*ACFNLength)], 
    ACFData[int(iTime*ACFNLength):int((iTime+1)*ACFNLength)], mode = "full"))

  ACF2 = np.average(ACF2, axis = 0)

  plt.rc('font',family='serif',size=16)
  # plt.rc('text',usetex=True)
  plt.rc('xtick',labelsize='small')
  plt.rc('ytick',labelsize='small')
  fig = plt.figure(figsize=(7,5.25))
  ax = fig.add_subplot(1,1,1)
  ax.plot(np.linspace(0., ACFtLength, num = int(ACFNLength)), ACF2[int(ACFNLength-1):],  color = "k")
  ax.set_ylabel('ACF of ACF')
  if sim.PairStyle == "Coul":
    ax.set_xlabel(r'$t \omega_p1$')
  else:
    ax.set_xlabel(r'$t/\tau$')
  fig.savefig(InputACF[:-4]+"_ACF_"+"tLength"+"_"+str(ACFtLength)+"_"+str(tStart)+"_"+str(tEnd)+".png")

def ChiSquare(InputFile, InputACF, tStart, tEnd, NBins):
  sim = simulation(InputFile)
  NStart = int(tStart/sim.tStep)
  NEnd   = int(tEnd/sim.tStep)
  
  ACFData = pd.read_csv(
  # file name
  InputACF,
  # skip header rows and the rows that label which divides the reps
  skiprows = list(range(NStart+2)),
  nrows = NEnd - NStart + 1, usecols = [1], delimiter = ' ', header=None
  ).to_numpy()
  
  ACFData = np.squeeze(ACFData)
  
  n, bins = np.histogram(ACFData, bins=NBins, density=False)
  Mean, SD = MeanSD(ACFData)
  zeromean = scipy.stats.norm.pdf([(bins[i]+bins[i+1]) / 2 for i in range(len(bins)-1)],0.0,SD)*(bins[1]-bins[0])*(NEnd - NStart + 1)
  
  stat, pval = scipy.stats.chisquare(n, zeromean, ddof = 1)
  print(stat, pval)

# unused codes

# codes for t-tests with specified sample interval in tau

# tStart = 50.
# tEnd   = 100.

# iStart = int(tStart/0.005)
# iEnd   = int(tEnd/0.005)

# SampInt = 5.

# RawData = pd.read_csv(
# # file name
# "r4_rho_1.0_T_1.0_Type1_VACF_All_NSample_384.txt",
# # skip header rows and the rows that label which divides the reps
# skiprows = list(range(iStart+2)),
# nrows = iEnd - iStart + 1, usecols = [1], delimiter = ' ', header=None
# ).to_numpy()

# mask = np.zeros(len(RawData), dtype=bool)
# mask[[i*int(SampInt/0.005) for i in range(int((tEnd-tStart)/SampInt))]] = True
# SelectData = RawData[mask]

# f1, ax1 = plt.subplots()
# ax1.plot(np.linspace(tStart, tEnd, num = int((tEnd - tStart)/SampInt)), SelectData,  color = "k")
# ax1.set_ylabel('VACF')
# ax1.set_xlabel(r'$t/\tau$')
# f1.savefig("VACF_"+str(tStart)+"_"+str(tEnd)+"_" +str(SampInt) +"int"+".png")

# f2, ax2 = plt.subplots()
# n, bins, patches = ax2.hist(SelectData, 50, density=True, stacked=True, facecolor='b', alpha=0.5, label="data with normalized freq")
# Mean, SD = MeanSD(SelectData)
# y = scipy.stats.norm.pdf(bins,Mean,SD)
# z = scipy.stats.norm.pdf(bins,0.0,SD)
# ax2.plot(bins, y, 'k-', label = "normal distribution by fitting")
# ax2.plot(bins, z, 'r--', label = "normal distribution w/ zero mean")
# ax2.set_xlabel('VACF')
# ax2.legend()
# f2.savefig("hist_VACF_"+str(tStart)+"_"+str(tEnd)+"_"+str(SampInt)+"int"+".png")

# tStat, pval = scipy.stats.ttest_1samp(SelectData, 0.0)
# print(tStat, pval)

# ---------------------------------------------------

# Mann Whitney U test

# tStart1 = 120.
# tEnd1   = 250.

# iStart1 = int(tStart1/0.005)
# iEnd1   = int(tEnd1/0.005)

# tStart2 = 270.
# tEnd2   = 400.

# iStart2 = int(tStart2/0.005)
# iEnd2   = int(tEnd2/0.005)

# RawData1 = pd.read_csv(
# # file name
# "r4_rho_1.0_T_1.0_Type1_VACF_All_NSample_384.txt",
# # skip header rows and the rows that label which divides the reps
# skiprows = list(range(iStart1+2)),
# nrows = iEnd1 - iStart1 + 1, usecols = [1], delimiter = ' ', header=None
# ).to_numpy()

# RawData2 = pd.read_csv(
# # file name
# "r4_rho_1.0_T_1.0_Type1_VACF_All_NSample_384.txt",
# # skip header rows and the rows that label which divides the reps
# skiprows = list(range(iStart2+2)),
# nrows = iEnd2 - iStart2 + 1, usecols = [1], delimiter = ' ', header=None
# ).to_numpy()

# tStat, pval = scipy.stats.mannwhitneyu(RawData1, RawData2, alternative="two-sided")
# print(tStat, pval)

# ----------------------------------------------------------

# bartlett test for equal variance

# tStart1 = 150.
# tEnd1   = 200.

# iStart1 = int(tStart1/0.005)
# iEnd1   = int(tEnd1/0.005)

# tStart2 = 350.
# tEnd2   = 400.

# iStart2 = int(tStart2/0.005)
# iEnd2   = int(tEnd2/0.005)

# SampInt = 5.

# RawData1 = pd.read_csv(
# # file name
# "r4_rho_1.0_T_1.0_Type1_VACF_All_NSample_384.txt",
# # skip header rows and the rows that label which divides the reps
# skiprows = list(range(iStart1+2)),
# nrows = iEnd1 - iStart1 + 1, usecols = [1], delimiter = ' ', header=None
# ).to_numpy()

# RawData2 = pd.read_csv(
# # file name
# "r4_rho_1.0_T_1.0_Type1_VACF_All_NSample_384.txt",
# # skip header rows and the rows that label which divides the reps
# skiprows = list(range(iStart2+2)),
# nrows = iEnd2 - iStart2 + 1, usecols = [1], delimiter = ' ', header=None
# ).to_numpy()

# mask1 = np.zeros(len(RawData1), dtype=bool)
# mask1[[i*int(SampInt/0.005) for i in range(int((tEnd1-tStart1)/SampInt))]] = True
# SelectData1 = np.squeeze(RawData1[mask1])

# mask2 = np.zeros(len(RawData2), dtype=bool)
# mask2[[i*int(SampInt/0.005) for i in range(int((tEnd2-tStart2)/SampInt))]] = True
# SelectData2 = np.squeeze(RawData2[mask2])

# tStat, pval = scipy.stats.bartlett(SelectData1, SelectData2)
# print(tStat, pval)
