# ShellModel

This is an overview of the various classes contained in this project:

Collection: contains a collection of beads  has sublcasses for a sphere, a cylindet, a disc, and a helix

Analyzer: Performs calculations of quantities like the viscous drag tensor under conditions of a certain tensor viscosity and taking into account (only) hydrodynamic interactions using methods developed by Teriado et. al.

ParticleArray: establishes a set of collection objects that can hydrodynamically "feel" each other, but can still move independently of each other. This class also contains code to simulate motion via an Euler-method timestepping approach.

ArrayMovie: takes a ParticleArray instance and performs several timesteps on it, recording the position and orientation infromation of the particles so that it can make an animated movie of the sedimentation.
