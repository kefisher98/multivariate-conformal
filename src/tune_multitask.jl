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
using Zygote

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
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")
include( path*"/src/modules/hypercube.jl")

# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------

function tune_multitask_hyperparameters( θ, ρ_set, s_set, tasks, xyz, settings, tune_seed ; file="tune_multitask.csv", j=1e-8, n_tune=(p=20,s=10), fold=(p=20,s=10) )

    # draw configuration indices
    Random.seed!(tune_seed);
    Sp = split( sample( xyz[tasks.p].inds, n_tune.p, replace=false  ), r(n_tune.p/fold.p) )
    Ss = split( sample( xyz[tasks.s].inds, n_tune.s, replace=false  ), r(n_tune.s/fold.s) )
    np = length(Sp)
    ns = length(Ss)
    np = 10#3
    ns = 10#3
    
    # initialize
    ε  = zeros( length(s_set), length(ρ_set) )

    for (ip,is) in Iterators.product(1:np,1:ns)
        T = ( p=Sp[ip], s=Ss[is] ) 
        C = ( p=vcat(1,Sp[setdiff(1:np,ip)]...), s=vcat(Ss[setdiff(1:ns,is)]...) )
        K, Ks, eCp, eCs, eTp, eTs = get_covariance( tasks ; C, T )

        for (ms, mρ) in Iterators.product( 1:length(s_set), 1:length(ρ_set) )
            #s = s_set[ms]
            s = 10.0^s_set[ms]
            ρ = ρ_set[mρ]
            ε[ms,mρ] += train( H(K;j), Ks, eCp, eCs, eTp, eTs ; C, T, θ, s, ρ  ) / ( np*ns)
        end
    end
    CSV.write( file, Tables.table( [ 0. ρ_set' ; s_set ε ] ) ) 
end


# ---------------------------------------------------------------
# ===============================================================
#
# Train model
#
# ===============================================================
# ---------------------------------------------------------------



function count_atoms(config)
    if     config < 329
        return 2
    elseif config < 549
        return 16
    elseif config < 659
        return 54
    elseif config < 714
        return 128
    elseif config < 1976
        return 63
    else
        return 215
    end
end

function get_covariance( tasks ; C, T  )

    # read features
    xCp, eCp = describe( tasks.p, C.p ; xyz, settings, type="energy" )
    xCs, eCs = describe( tasks.s, C.s ; xyz, settings, type="energy" )
    xTp, eTp = describe( tasks.p, T.p ; xyz, settings, type="energy" )
    xTs, eTs = describe( tasks.p, T.s ; xyz, settings, type="energy" )
    nap, nas = count_atoms.(T.p), count_atoms.(T.s)

    # training covariance and test-training covariance
    nC, nT   = ℓ(C.p)+ℓ(C.s), ℓ(T.p)+ℓ(T.s)
    K        = covariance(  [xCp;xCs;xTp;xTs], 1:nC,        1:nC       ; C=energy )
    Ks       = covariance(  [xCp;xCs;xTp;xTs], 1:nC,        nC.+(1:nT) ; C=energy )

    return (K+K')/2, Ks, eCp, eCs, eTp, eTs
end

function train( K, Ks, eCp, eCs, eTp, eTs ; C, T, θ, s, ρ  )

    # separate indices
    nCp, nCs, nTp, nTs = ℓ(C.p), ℓ(C.s), ℓ(T.p), ℓ(T.s)

    # construct multitask training covariance
    nCp, nCs, nTp, nTs               = ℓ(C.p), ℓ(C.s), ℓ(T.p), ℓ(T.s)
    M  = [   ones(nCp,nCp)          ρ*ones(nCp,nCs) ;
           ρ*ones(nCs,nCp)  (ρ^2 + s)*ones(nCs,nCs) ]
    Ms = [   ones(nCp,nTp)          ρ*ones(nCp,nTs) ;
           ρ*ones(nCs,nTp)  (ρ^2 + s)*ones(nCs,nTs) ]

    # test
    μ = (Ks.*Ms)' * (( K.*M + θ*I ) \ [eCp;eCs])
    
    return rmse(μ,[eTp;eTs])
end

#-------

covariance( X, aa, bb ; C ) = vcat([ hcat([ C(X[a], X[b])  for b in bb ]...)  for a in aa ]...)
energy( A, B ;  ζ=4  )      = sum((A*B').^ζ)


# ---------------------------------------------------------------
# ===============================================================
#
# User defined settings 
#
# ===============================================================
# ---------------------------------------------------------------

equilibrium = Dict( 2  =>225,
                    16 =>329,
                    54 =>549,
                    128=>659,
                     63=>1876,
                    215=>1798 )

diamonds     = 225:713
vacancies    = 1876:2086
tune_seed    = 1
files        = [ path*"/data/configurations/GAP_silicon.xyz", path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy","dft_energy"]
derivatives  = ["dft_force","dft_force"]
stresses     = ["dft_virial","dft_virial"]
tasks        = ["dia","vacancy"]
primary      = setdiff( diamonds,   collect(keys(equilibrium)) )
secondary    = setdiff( vacancies,  collect(keys(equilibrium)) )
ρ_set        = collect(0.05:0.05:1.)
s_set        = collect(-2:0.5:2)
θ            = 1e-8


#   2 atoms:  225:328
#  16 atoms:  329:548
#  54 atoms:  549:658
# 128 atoms:  659:713
#  63 atoms vacancy: 1876:1975
# 215 atoms vacancy: 1976:2086
# divacancy: 1798:1875

# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [primary,secondary], stresses ; tasks  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )
tasks        = (p=tasks[1], s=tasks[2])

file = path*"/data/hyperparameters/tune_multitask.csv"
tune_multitask_hyperparameters( θ, ρ_set, s_set, tasks, xyz, settings, tune_seed ; file ) 

