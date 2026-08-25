using LinearAlgebra
using Random
using CSV
using DataFrames
using Distributions
using Tullio
using SpecialFunctions
using StaticArrays
using NeighbourLists
using Optim
using LineSearches

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

# import local files
path  = @__DIR__
path  = path[1:end-4]
include( path*"/src/modules/auxiliary.jl")
include( path*"/src/modules/material_settings.jl")
include( path*"/src/modules/quip_descriptors.jl")

# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------

# Runtime (n_tune=20, fold=20) : 96.041420 seconds

function tune_energy_hyperparameter( params, tune_seed, xyz, settings ; task="dia", file="energy_tuning.csv", j=1e-12, n_tune=20, fold=20 )

    # draw configuration indices
    Random.seed!(tune_seed);
    S  = split( sample( xyz[task].inds, n_tune, replace=false  ), r(n_tune/fold) )
    nS = length(S)
    nP = length(params)

    # initialize 
    ε  = zeros(nP)

    for i in 1:nS
        T = S[i]
        C = vcat(1,S[setdiff(1:nS,i)]...)
        K, Ks, eC, eT = covariance( task, xyz ; settings, C, T )

        for m in 1:nP
            θ     = 10.0^(params[m])
            μ     = Ks' * (( H(K;j) + θ*I) \ eC )
            ε[m] += rmse(μ,eT) / nS
        end
    end
    CSV.write( file, Tables.table( [params ε] ) ) 
end

#-------

covariance( X, aa, bb ; C ) = vcat([ hcat([ C(X[a], X[b])  for b in bb ]...)  for a in aa ]...)
energy( A, B ;  ζ=4  )      = sum((A*B').^ζ)

#-------

function covariance( task, xyz ; settings, C, T)
    xC, eC = describe( task, C ; xyz, settings, type="energy" )
    xT, eT = describe( task, T ; xyz, settings, type="energy" )
    nC, nT = (ℓ(C), ℓ(T))

    # training covariance and test-training covariance
    Ke     = covariance(  [xC; xT], 1:nC,       1:nC ;       C=energy )
    Ke_s   = covariance(  [xC; xT], 1:nC,       nC.+(1:nT) ; C=energy )

    return Ke, Ke_s, vcat(eC...), vcat(eT...)
end


# ---------------------------------------------------------------
# ===============================================================
#
# Set up test
#
# ===============================================================
# ---------------------------------------------------------------

equilibrium = Dict( 2  =>225,
                    16 =>329,
                    54 =>549,
                    128=>659 )

tune_seed    = 1
files        = [ path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
task         =  "dia"
diamonds     = 225:713
inds         = setdiff( diamonds, collect(keys(equilibrium)) )
params       = collect(1:-1:-12)


# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [inds], stresses  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )

file = path*"/data/hyperparameters/tune_energy.csv"
tune_energy_hyperparameter( params, tune_seed, xyz, settings ; task, file )
