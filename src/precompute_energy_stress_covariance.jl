using LinearAlgebra
using Random
using CSV
using DataFrames
using Distributions
using Tullio
using SpecialFunctions
using NeighbourLists
using StaticArrays

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

# import local files
path  = @__DIR__
path  = path[1:end-4]
include( path*"/src/modules/auxiliary.jl")
include( path*"/src/modules/material_settings.jl")
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")
include( path*"/src/modules/inner_kernel.jl")
include( path*"/src/modules/quip_descriptors.jl")

# ------------------------
# ==============================================================================
# prediction manager
# ==============================================================================
# -----------------------


function get_descriptors( C ; type )
    files        = [ path_basic*"/data/configurations/GAP_silicon.xyz"]
    key          =   path_basic*"/data/configurations/key_structure.csv"
    quantities   = ["dft_energy"]
    derivatives  = ["dft_force"]
    stresses     = ["dft_virial"]
    task         =  "dia"
    inds         = 329:713

    # run experiment
    xyz          = set_xyz( files, quantities, derivatives, key, inds, stresses  )
    descriptor   = set_SOAP( ; species=["Si"], r_cut=5.0 )
    settings     = set_settings( ;  descriptor )

    return describe( task, C ;  xyz, settings, type )
end


# ------------------------
# ==============================================================================
# Implement tests
# ==============================================================================
# -----------------------

# read configuration indices
path_basic = path
path       = string(   path, "/data/covariance/" )
C          = CSV.read( path*"stress_C.csv" ,    DataFrame)[:,1]



# energy and solitary energy to force
X, eC, fC  = get_descriptors( C ; type=string(stress) )
nC = length(C)
Ke = covariance(  X, 1:nC, 1:nC  ;  C=energy )
S  = covariance(  X, 2:nC, 1     ;  C=stress_energy )
X  = nothing


# construct Knm
Kse = [ S Matrix(CSV.read( path*"stress_energy_compiled_K.csv", DataFrame )) ]
K   = [ Ke  Kse'  ;  Kse  Matrix(CSV.read( string( path, "stress_compiled_K.csv") , DataFrame)) ]

CSV.write( string( path, "stress_compiled_K.csv"), Tables.table(K))
