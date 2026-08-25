# test_relaxation_quip.jl
# Test geometry optimization with SOAP descriptors

using LinearAlgebra
using Random
using Tullio
using Optim
using LineSearches
using Printf
using Statistics
using CSV
using DataFrames
using Statistics
using Distributions
using PyCall
ase = pyimport("ase.io")
atoms = pyimport("ase")

path  = @__DIR__
path  = path[1:end-4]
include(path * "/src/modules/auxiliary.jl")
include(path * "/src/modules/quip_descriptors.jl")
include(path * "/src/modules/material_settings.jl")

# ============================================================================
# Outer loop
# ============================================================================

function tune_difference_hyperparameters( θ, step_set, task, xyz, settings, tune_seed ; n_tune=20, n_test=10, file="tune_steps.csv" )

    # draw configuration indices
    Random.seed!(tune_seed);
    S = sample( xyz[task].inds, n_tune, replace=false  )

    # read data 
    configs, xS, K, eS, vS = retrieve( S ; xyz, task )
 
    ε = zeros(length(step_set),6)
    for T in sample( 1:length(S), n_test, replace=false )
        C  = setdiff(1:length(S),T)
        ε += test_difference( step_set, configs[T], xS[C], K[C,C], eS[C], vS[T], θ ) / n_test
    end
    CSV.write( file, Tables.table( [step_set ε] ) )
end

function retrieve( select ; xyz, task )
    desc     = SOAPDescriptor(species=["Si"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5)
    configs  = ase.read( xyz[task].file, index=":" )[select]
    energies = [ c.info[ xyz[task].qoi]  for c in configs ]
    virials  = [ reshape(c.info[  xyz[task].s],:,1)[[1,5,9,2,3,6]]    for  c in configs ]

    nS = length(select)
    xS = grad_describe.(configs, Ref(desc))
    K  = covariance( last.(xS), 1:nS, 1:nS ; C=energy )

    return configs, xS, K, energies, virials
end

covariance( X, aa, bb ; C ) = vcat([ hcat([ C(X[a], X[b])  for b in bb ]...)  for a in aa ]...)
energy( A, B ;  ζ=4  )      = sum((A*B').^ζ)


# ============================================================================
# Relaxation
# ============================================================================

#  Kernel functions (energy-only, simpler version)
function ref_energy(A, B; ζ=4)
    return sum((last(A) * last(B)') .^ ζ)
end

function ref_force(A, B; ζ=4)
    δ = ζ * (last(A) * last(B)') .^ (ζ - 1)
    @tullio fe[i, j] := δ[a, b] * last(B)[b, r] * first(A)[a, i, j, r]
    return fe
end

function relax(cell, G0; s, pbc, desc, xC, w, v=1 )
    nC = length(xC)
    function perturbed_energy(G)
        a = atom_create(G, s, cell, pbc)
        x = grad_describe(a, desc)
        E = sum(w[b] * ref_energy(x, xC[b]) for b in 1:nC)
        return v * E
    end
    function perturbed_force!(F, G)
        a = atom_create(G, s, cell, pbc)
        x = grad_describe(a, desc)
        fill!(F, 0.0)
        for b in 1:nC
            F .+= w[b] * ref_force(x, xC[b])
        end
        F .*= v
    end
    sol = optimize( perturbed_energy, perturbed_force!, G0, LBFGS(linesearch=LineSearches.BackTracking()),
                    Optim.Options(   g_tol=1e-2,
                                     x_abstol=1e-2,
                                     iterations=100,
                                     show_trace=false,
                                     extended_trace = true,
                                     store_trace=true,
                                     show_every=5 ))
    return sol
end


# ============================================================================
# Relaxation manager
# ============================================================================

function perturb_relax( G_perturbed, s, cell, pbc, xC, w, desc )
    #= 
    nC = length(xC)
    a  = atom_create(G_perturbed, s, cell, pbc)
    x  = grad_describe(a, desc)
    return sum(w[b] * ref_energy(x, xC[b]) for b in 1:nC)
    =#

    sol = relax(cell, G_perturbed; s, pbc, desc, xC, w )
    gi  = [ iter.g_norm  for iter in Optim.trace(sol) ]
    pt  = nothing
    for j in 2:length(gi)
        k_set=[]
        for k in 1:j-1
            if gi[j]/gi[k] > 5
                push!(k_set,k)
            end
        end
        if length(k_set)>0
            pt = k_set
            break
        end
    end

    if isnothing(pt)
        G_final = Optim.minimizer(sol)
        E_final = Optim.minimum(sol)
    else
        leap    = [ gi[k+1]/gi[k]   for k in pt ]
        stop    = pt[findfirst( leap.>1 )]
        G_final = Optim.trace(sol)[stop].metadata["x"]
        E_final = Optim.trace(sol)[stop+1].value
    end     

    return E_final
end

function test_difference( step_set, test, xC, K, eC, vT, θ ; j=1e-8 )

    # Build energy-only kernel and solve
    w = ( H(K;j) + θ*I ) \ eC 

    # Get test configuration
    G0, s, cell, pbc = atom_extract(test)
    desc     = SOAPDescriptor(species=["Si"], r_cut=5.0, n_max=8, l_max=6, sigma=0.5)

    #####################################################################################
    ε = zeros(length(step_set),6)
    for z in 1:length(step_set)
        Δ      = step_set[z]
        stress = zeros(6)

        for i in 1:3
            u      = Δ*diagm(ej(i)) 
            ep     = perturb_relax( G0*(I+u), s, cell*(I+u), pbc, xC, w, desc )
            em     = perturb_relax( G0*(I-u), s, cell*(I-u), pbc, xC, w, desc )
            ε[z,i] = ( (ep - em)/(2Δ) + vT[i] )/vT[i]
        end
        loc = 4
        for i in 1:2
           for j in i+1:3
                u        = 0.5*Δ*( ej(i)*ej(j)' + ej(j)*ej(i)' )
                ep       = perturb_relax( G0*(I+u), s, cell*(I+u), pbc, xC, w, desc ) 
                em       = perturb_relax( G0*(I-u), s, cell*(I-u), pbc, xC, w, desc )         
                ε[z,loc] = ((ep - em)/(2Δ) + vT[loc])/vT[loc]
                loc      += 1
            end
        end
    end

    return ε
end


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
                    128=>659 )

tune_seed    = 1
files        = [ path*"/data/configurations/GAP_silicon.xyz", path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
tasks        = ["dia"]
diamonds     = 225:713
inds         = setdiff( diamonds, collect(keys(equilibrium)) )
step_set     = [ 0.0001, 0.001, 0.005, 0.0063, 0.0075, 0.0087, 0.01, 0.013, 0.025, 0.037, 0.05, 0.1, 1. ]
θ            = 1e-8




#   2 atoms:  225:328
#  16 atoms:  329:548
#  54 atoms:  549:658
# 128 atoms:  659:713
#  63 atoms vacancy: 1876:1975
# 215 atoms vacancy: 1976:2086
# divacancy: 1798:1875

# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [inds], stresses ; tasks  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )


file = path*"/data/hyperparameters/tune_finite_diff.csv"
tune_difference_hyperparameters( θ, step_set, tasks[1], xyz, settings, tune_seed ; file )
