#-------------------------------------------------------------------
#
#                 OCP_BIM_Diff_PyL_1.0/ReadInput.py
#    Tianpu Zhao (TPZ), tz1416@ic.ac.uk, pacosynthesis@gmail.com
#
#-------------------------------------------------------------------
# 
# Description: 
# This module contains the simulation class that reads input file
# that contains the essential simulation parameters for either one
# component fluid or binary mixture, interacting via an inverse 
# polynomial potential or long-range Coulombic potential. It also
# calculates several quantities for analysis. All the quantities in
# the simulations are in LJ dimensionless units, as specified in [2], 
# and all the expressions for derived variables can also be found in [1].
# 
# Module prerequisites:
# The versions below are the ones that I use when developing the 
# code. Earlier/later version may or may not work. 
# python: 3.8.5
# numpy: 1.19
#
# Here are brief descriptions for the functions/classes in this
# module.
#
# species:
# class for species (type of particles) information
# 
# simulation:
# class for the simulation parameters and useful parameters for 
# analysis.
#
#-------------------------------------------------------------------
#
# [1]: unit_conversions.pdf
# [2]: https://lammps.sandia.gov/doc/units.html
# [3]: Shaffer et al. PRE 95 013206 (2017)
# [4]: https://lammps.sandia.gov/doc/fix_ave_correlate.html
#
#-------------------------------------------------------------------
#
# 2020.08.08 Created                                 TPZ
# 2020.09.04 Modified                                TPZ
#            (added new classes species and simulations
#             and reading input.txt)
# 2020.09.07 Modified                                TPZ
#            (completed the class species and simulations
#            and the function for reading input)
# 2020.09.19 Modified                                TPZ
#            (edited comments)
# 2021.03.31 Modified                                TPZ
#            (upgraded to calculate diffusion coeffs on-
#            the-fly)
# 
#-------------------------------------------------------------------

import numpy as np

# objects explained for species class:
# mass: mass
# charge: charge number
# number: particle number in the simulation box
# molfrac: molar (number) fraction in the mixture
# massfrac: mass (density) fraction in the mixture
class species:
  def __init__(self, mass, charge, number):
    self.mass     = mass
    self.charge   = charge
    self.number   = number
    self.molfrac  = 0.
    self.massfrac = 0.
  def AssignMolFrac(self, molfrac):
    self.molfrac  = molfrac
  def AssignMassFrac(self, massfrac):
    self.massfrac = massfrac

# objects explained for simulation class:
#
# Simulation type:
# species: a list of species class, including the information for each
#          species. The first species must be the lightest.
# mode: string, "binary" or "pure", specify whether the simulation is 
#       running for binary mixture or pure fluid.
# PairStyle: string, "Coul" or "poly", specify the interaction potential,
#            "Coul" for long-range Coulombic potential that requires a 
#            lattice sum (Ewald, PPPM, PME, etc.), "poly" for inverse
#            polynomial potential with a cutoff, developed by Simran
#            Chowdhry.
# NDim: integer, 2 or 3, number of dimensions.
#
# Species info:
# NReps : integer, number of time origins for VACF (and also number of 
#         VACF computes invoked for each species). It can also be 
#         interpreted as the total number of repeating the VACF computes,
#         hence the name NReps.
# NTotal, MTotal: total number and mass of particles in the system.
# NSpecies: number of species in the system.
# 
# Simulation parameters:
# tStep: float, timestep measured in tau for PairStyle "poly", in plasma 
#        time of species 1 for PairStyle "Coul"
# NReps: The number of repetitions of VACF computes in a single trajectory,
#        it is also the number of time origins.
#
# the following variables that begin with t are measured in time units
# (float), those begin with N are measured in timesteps (integer):
# t/NEqm: simulation time for equilibration stage
# t/NInt: The time between each VACF origin
# t/NLength: The length (in time) of a single VACF and PCACF
# t/NFreq: The interval (in time) between averages of PCACF to be 
#          evaluated and reported, same as Nfreq in [4]
# NEvery: The number of timesteps between subsequent VACF and PCACF 
#         calculations,can be interpreted as "sample VACF/PCACF every 
#         NEvery steps" 1 means VACF is calculated for every step.
#
# ASCII demonstration:
#               / |___________|___________________________________________|
#              |    Eqm stage :            Prod stage                     :
#              |      tEqm    :                                           : 
# 1 Trajectory |            / |________________|                          :
# labelled by <  NReps (=4)|  :   tLength (VACF)                          :
# iTraj        | of time  <   :        |________________|                 :
#              | origins   |  :        :        |________________|        :  
#              | for VACF   \ :        :                 |________________| 
#              |              \____ ___/                                  :
#              |              :    V                                      :
#              |              :   tInt (VACF)                             :
#              |            / |________________|                          :
#              |           |  :    tLength            |________________|  :
#              | for PCACF<   :(PCACF recorded)       :                   :
#              |           |  :                       :                   :
#               \           \ \___________ ___________/                   :
#                                         V
#                                       tFreq 
#                             (interval of PCACF recorded)
#
# File names:
# EqmLogStem, ProdLogStem, FileStem: all strings, the stem of the output
# files.
# File names for
# setup and equilibrium stage log: {EqmLogStem}_{iTraj}.txt
# production stage log           : {ProdLogStem}_{iTraj}.txt
# VACF and self-diffusion file   : {FileStem}_{iTraj}_Dii_Type{iType}_{iRep}.txt
# PCACF file                     : {FileStem}_{iTraj}_D12.txt
#
# cutoff: float, for different PairStyle:
# "poly": in sigma, potential cutoff in sigma
# "Coul": in Wigner-Seitz radius, cutoff between direct and inverse space
# potential treatment
#
# PairStyle-specific parameters:
# For "Coul":
# Gamma: float, coupling parameter, depending on the GammaMode
# GammaMode: Gamma0 for eq. (3) in [3], Gamma for eq. (2) in [3], when simulating
#            OCP, Gamma0 is equivalent to the Gamma used in OCP. Both of them
#            will be calculated in the end.
# NGrids: integer, number of grids on each dimension for PPPM scheme
# aWS: float, Wigner-Seitz radius
# omega_p: float, mixture plasma frequency, eq. (36) in [3]
# omega_p1: float, plasma frequency of the lighter species
# numden: float, total number density
# rho: float, total mass density
#
# For "poly":
# numden: float, total number density
# polyCoeff: list with float, coefficients of polynomial terms in descending 
#            order, begin with the constant term (and following with the terms 
#            r^-1, r^-2, etc.), essentially the A, B, C, D, E,... are specfied 
#            for the follwing interaction potential
#            u(r) = Z_i*Z_j*(A + B/r + C/r^2 + D/r^3 + E/r^4 + ...) 
#            Z_i and Z_j are charges of species i and j
#
class simulation:
  def __init__(self, InputFile):
    with open(InputFile, "r") as f:
      lines = f.read().split("\n")

      # mode, pair style, dimension
      self.mode      = lines[21].strip()
      self.PairStyle = lines[24].strip()
      self.NDim      = int(lines[27])
      if not (self.NDim == 2 or self.NDim == 3):
        raise Exception("the simulation dimension must be 2 or 3")

      # species information
      if self.mode == "pure":
        self.species = []
        word = lines[32].split()
        self.species.append(species(float(word[0]), float(word[1]), int(word[2])))
      if self.mode == "binary":
        self.species = []
        word1 = lines[32].split()
        word2 = lines[36].split()
        self.species.append(species(float(word1[0]), float(word1[1]), int(word1[2])))
        self.species.append(species(float(word2[0]), float(word2[1]), int(word2[2])))
        if self.species[0].mass != min([iSpecies.mass for iSpecies in self.species]):
          raise Exception("Species 1 mass must be the smallest")
      self.NTotal = sum([iSpecies.number               for iSpecies in self.species])
      self.MTotal = sum([iSpecies.number*iSpecies.mass for iSpecies in self.species])
      self.NSpecies = len(self.species)
      self.GetComposition()

      # simulation times
      self.tStep     = float(lines[42])
      self.tEqm      = float(lines[45])
      word = lines[57].split()
      self.NReps,   self.tInt,      self.tLength,   self.tFreq,     self.NEvery = \
      int(word[0]), float(word[1]), float(word[2]), float(word[3]), int(word[4])
      
      # convert time to steps
      self.NEqm    = self.ttoN(self.tEqm)
      self.NInt    = self.ttoN(self.tInt)
      self.NLength = self.ttoN(self.tLength)
      self.NFreq   = self.ttoN(self.tFreq)
      if not (self.NLength % self.NEvery == 0):
        raise Exception("NLength need to be multiples of NEvery")
      if not (self.NFreq % self.NEvery == 0):
        raise Exception("NFreq need to be multiples of NEvery")
      
      # file names
      word = lines[68].split()
      self.EqmLogStem, self.ProdLogStem, self.FileStem = \
      word[0].strip(), word[1].strip() , word[2].strip()

      # cutoff
      self.cutoff    = float(lines[75])

      # pair style specific arguments
      if self.PairStyle == "Coul":
        word = lines[84].split()
        if word[0] == "Gamma0":
          self.Gamma0,    self.NGrids  =\
          float(word[1]), int(word[2])
          self.Gamma0toGamma()
          self.CalcPlasmaParameters()
        elif word[0] == "Gamma":
          self.Gamma,     self.NGrids  =\
          float(word[1]), int(word[2])
          self.GammatoGamma0()
          self.CalcPlasmaParameters()
        else:
          raise Exception("GammaMode need to be either Gamma or Gamma0")
      elif self.PairStyle == "poly":
        self.numden = float(lines[88])
        word = lines[93].split()
        self.polyCoeff = [float(iCoeff) for iCoeff in word]
      else:
        raise Exception("PairStyle need to be Coul or poly, other styles are not yet supported")

      # output file directory
      self.dir = lines[17].strip()
  
  # convert time into timestep
  def ttoN(self, tval):
    Nval = int(tval/self.tStep)
    return Nval
  
  # get the mole and mass fraction of each species
  def GetComposition(self):
    for iSpecies in self.species:
      iSpecies.AssignMolFrac(iSpecies.number / self.NTotal)
      iSpecies.AssignMassFrac(iSpecies.number*iSpecies.mass / self.MTotal)
  
  # number average of a quantity A for each species, A need to be a vector with length NSpecies
  def NumAvg(self, A):
    if self.NSpecies != np.size(A):
      raise Exception("The value to be averaged must have the same length as number of species")
    avgA = np.dot([iSpecies.molfrac for iSpecies in self.species], A)
    return avgA
  
  # calculate various parameters of plasma, and overall density 
  def CalcPlasmaParameters(self):
    # number-averaged charge number
    avgZ     = self.NumAvg([iSpecies.charge for iSpecies in self.species])
    # number-averaged mass
    avgMass  = self.NumAvg([iSpecies.mass for iSpecies in self.species])
    # Wigner-Seitz radius
    aWS           = 1./self.Gamma0
    self.aWS      = aWS
    # mixture plasma frequency, eq. (36) in [3]
    self.omega_p  = (3.                          / ((aWS*aWS*aWS) * avgMass      )) ** (1./2.) * avgZ
    # plasma frequency of the lighter species
    self.omega_p1 = (3. * self.species[0].molfrac/ ((aWS*aWS*aWS) * self.species[0].mass)) ** (1./2.) * self.species[0].charge
    # total number density
    self.numden   = 3./(4.*np.pi*aWS*aWS*aWS)
    # total mass density
    self.rho      = avgMass*self.numden
  
  # Gamma_0 as defined by eq. (3) in [3], from Gamma bar defined by eq. (2) in [3]
  def Gamma0toGamma(self):
    avgZ = self.NumAvg([iSpecies.charge for iSpecies in self.species])
    self.Gamma  = self.Gamma0 * (avgZ ** (1./3.) * self.NumAvg([iSpecies.charge ** (5./3.) for iSpecies in self.species]))
  # Gamma bar as defined by eq. (2) in [3], from Gamma_0 defined by eq. (3) in [3]
  def GammatoGamma0(self):
    avgZ = self.NumAvg([iSpecies.charge for iSpecies in self.species])
    self.Gamma0 = self.Gamma  / (avgZ ** (1./3.) * self.NumAvg([iSpecies.charge ** (5./3.) for iSpecies in self.species]))

        