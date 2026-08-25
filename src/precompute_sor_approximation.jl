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
C          = CSV.read( path*"force_C.csv" ,    DataFrame)[:,1]
sor        = CSV.read( path*"force_sor.csv",   DataFrame)[:,1] 
mol_sz     = CSV.read( path*"force_mol_sz.csv",DataFrame)[2:end,1]


# energy and solitary energy to force
X, eC, fC  = get_descriptors( C ; type=string(force) )
nC = length(C)
Ke = covariance(  X, 1:nC, 1:nC  ;  C=energy )
S  = covariance(  X, 2:nC, 1     ;  C=force_energy )
X  = nothing

# obtain m indices 
m       = vcat([ sum([0;mol_sz[1:i-2]]) .+ (1:mol_sz[i-1])  for i in sor[2:end] ]...)

# construct Knm
Kfe = [ S Matrix(CSV.read( path*"force_energy_compiled_K.csv", DataFrame )) ]
Knm = [ Ke[:,sor]  Kfe[m,:]'  ;  Kfe[:,sor]  Matrix(CSV.read( string( path, "force_Knm_", length(sor), ".csv") , DataFrame)) ]
Kfe = nothing

# smaller products
v = 1e5
σ²= 1e-3
M = v*Knm'*Knm*v + σ²*v*Knm[[sor;m],:]
y = v*Knm'*[eC ; -vcat(fC[2:end]...)]

name = string( path, "force_SOR_", length(sor), "_", length(C) )
CSV.write( name*".csv",    Tables.table(M) )
CSV.write( name*"_w.csv",  Tables.table( (M\y) ) )


