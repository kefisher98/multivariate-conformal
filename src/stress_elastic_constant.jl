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
include( path*"/src/modules/inner_kernel.jl" )
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")


# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------



function elastic_constant_from_stress( s, xyz, settings, loc_id, sz; 
                                       seed,
                                       h=0.05, task="dia", α_set=collect(0.4:0.05:0.95),
				                       n_tune=20, fold=20, tune_seed=1, tune_set, 
                                       σ²=[1e-8,1e-8], v=1.0, vE=1.0,
				                       core=[10], target=[175], energy_core=[300], j=1e-8  )

    # identify tuning configurations to set aside
    Random.seed!(tune_seed);
    LO  = split( sample( tune_set, n_tune, replace=false  ), r(n_tune/fold) )

    # draw configuration indices
    Random.seed!(seed);
    C = [1 ; sample( setdiff( xyz[task].inds[1], [s;LO]     ), maximum(core)-1,                      replace=false  )] # train
    T =      sample( setdiff( xyz[task].inds[2], [s;C;LO]   ), maximum(target),                      replace=false  )
    E = [C ; sample( setdiff( xyz[task].inds[3], [s;C;T;LO] ), maximum(energy_core)-minimum(core),   replace=false  )]  # EC prediction

    # iterate over training set sizes
    for (nC,nT,nE) in Iterators.product(core,target,energy_core)
        # train GP
        K_file=path*"/data/covariance/stress_compiled_K.csv"
        w, μ, Σ, xC, K, sC, sT, err, ρ = train_stress( task, xyz ; settings, C=C[1:nC], T=T[1:nT], σ², v, K_file )

        # train GP for relaxation
        wE, xE = train_energy( task, xyz ; settings, C=E[1:nE],  σ², v=vE  )

        # vanilla conformal scores
        cw  = [ component_wise( μ, Σ2σ(Σ), sT ; α ) for α in α_set ]
	    bf  = [ component_wise( μ, Σ2σ(Σ), sT ; α, bonf=true ) for α in α_set ]

        # multivariate conformal scores
        μ, Σ, sT =   extract(  μ, Σ, sT ; nT, nCal=nT*2 )
        mv       = [ multivar( μ, Σ, sT ; α )  for α in α_set ]
        
        # predict elastic constant
        sd   = settings.descriptor
        desc = SOAPDescriptor(  species=sd.species, n_max=sd.n_max, l_max=sd.l_max , sigma=sd.sigma, r_cut=sd.r_cut )
            
        # predict energies at perturbed geometries
        KS, errC, KC, errM, KM, full, reduced, i, Vm  = perturb_relax( s, w, xC, K, wE, xE ; desc, xyz, task, v, vE, σ², h, nC )

        # apply conformal
        cw_out = predict_componentwise.( first.(cw) ; KS, errC, KC, errM, KM, i, Vm, h, seed  )
	    bf_out = predict_componentwise.( first.(bf) ; KS, errC, KC, errM, KM, i, Vm, h, seed  )
        mv_out = predict_multivariate.(  first.(mv) ; KS, errC, KC, errM, KM, i, full, reduced, seed  )
	    report( loc_id, sz, cw_out, mv_out, bf_out, nC, nT, config, h  )
    end
end


# ---------------------------------------------------------------
# ===============================================================
#
# Report
#
# ===============================================================
# ---------------------------------------------------------------

function report( loc_id, sz, cw, mv, bf, nC, nT, config, h )
    path_cf = string( path, "/data/stress_elastic_constant/ec_stress_config", sz,  "_", h, "_nC", nC, "_nT", nT, "_", loc_id, "_conformal.csv" )
    h_cf    = [ :seed, :cw_alpha, :ec1, :ec2, :ec3, :ec4, :ec5, :ec6, :bm, 
                :cw_stress_volume, :cw_ec_volume, :cw_ec_volume_red,  :cw_fullcov, :cw_redcov, :cw_cwfullcov, :cw_cwredcov, :cw_Mvol, :cw_Mcov,
                :mv_stress_volume, :mv_ec_volume, :mv_ec_volume_red,  :mv_fullcov, :mv_redcov, :mv_cwfullcov, :mv_cwredcov, :cw_fullcov_upper, :cw_redcov_upper, :mv_Mvol, :mv_Mcov  ]
    addToFile( DataFrame( [hcat(cw...)'  hcat(mv...)'], h_cf ), path_cf )


    path_bf = string( path, "/data/stress_elastic_constant/ec_stress_config", sz,  "_", h, "_nC", nC, "_nT", nT, "_", loc_id, "_bonf.csv" )
    h_bf    = [ :seed, :cw_alpha, :ec1, :ec2, :ec3, :ec4, :ec5, :ec6, :bm,
                :cw_stress_volume, :cw_ec_volume, :cw_ec_volume_red,  :cw_fullcov, :cw_redcov, :cw_cwfullcov, :cw_cwredcov, :cw_Mvol, :cw_Mcov]
    addToFile( DataFrame( hcat(bf...)', h_bf ), path_bf )
end




function extract(m, C, y; skip=9, nT, nCal)
    inds = split(1:ℓ(m),skip)
    pair = [ sample(1:ℓ(inds), 2, replace=false) for _ in 1:nCal ]
    iCal = [ [ inds[p[1]]; inds[p[2]]]  for p in pair ]

    return [m[i] for i in iCal], [C[i,i] for i in iCal  ], [y[i] for i in iCal]
end


# ---------------------------------------------------------------
# ===============================================================
#
# Training
#
# ===============================================================
# ---------------------------------------------------------------

# TODO check that xC tuple matches downstream implementations

function train_stress( task, xyz ; settings, C, T, σ²=[0.001,0.01], v=10000.0, K_file, shift=327, skip=221  )

    # read in covariance
    s = [ C[1] ; C[2:end].-shift ; T.-shift ; [ (skip + 9(i-shift-2)) .+ (1:9)  for i in [C[2:end];T] ]...  ]
    K = Matrix( CSV.read( K_file, DataFrame))[s,s]

    # indices
    nC, nT = ℓ(C), ℓ(T)
    train  = [      1:nC  ; (nC+nT).+(1:9(nC-1))     ]
    test   = [ nC.+(1:nT) ; (nC+nT+9(nC-1)).+(1:9nT) ] 

    # construct descriptors
    xC, eC, sC = describe(  task, C ; xyz, settings, type="stress" )[[1,2,4]]
    eT, sT     = read_data( task, T ; xyz, settings )[[1,3]]
    sC, sT     = vcat(sC[2:end]...), vcat(sT...)

    # train GP
    η = [ σ²[1]*ones(nC);  σ²[2]*ones(9(nC-1)) ]
    w = (v*K[train,train] + diagm(η)) \ [eC; -sC]
    μ =  v*K[test, train]*w
    Σ = v*K[test,test]  -  v^2*K[test,train] * ( (v*K[train,train] + diagm(η)) \ K[train,test]  ) 

    # report error
    err = [ rmse(μ[1:nT],eT), rmse(-μ[nT+1:end],sT)    ]
    ρ   = [  cor(μ[1:nT],eT),  cor(-μ[nT+1:end],sT)[1] ]

    return w, -μ[nT+1:end], Σ[nT+1:end,nT+1:end], xC, K[train,train], sC, sT, err, ρ
end

function train_energy( task, xyz ; settings, C,  σ², v=10000.0  )

    # construct covariance structures
    xC, eC = describe( task, C ; xyz, settings, type="energy" )
    K      = covariance( [(0,x) for x in xC], 1:ℓ(C), 1:ℓ(C)   ;   C=energy )
    w      = (v*H(K) + σ²[2]*I) \ eC

    return w, xC
end

# ===============================================================
#
# Relaxation
#
# ===============================================================
# ---------------------------------------------------------------

function ref_energy(A, B; ζ=4)
    return sum((last(A) * B') .^ ζ)
end

#-----

function ref_force(A, B; ζ=4)
    δ = ζ * (last(A) * B') .^ (ζ - 1)
    @tullio fe[i, j] := δ[a, b] * B[b, r] * first(A)[a, i, j, r]
    return fe
end

#-----

function relax(cell, G0; s, pbc, desc, xE, wE, vE)
    nE = length(xE)

    function perturbed_energy(G)
        a = atom_create(G, s, cell, pbc)
        x = grad_describe(a, desc)
        E = sum(wE[b] * ref_energy(x, xE[b]) for b in 1:nE)
        return vE * E
    end

    function perturbed_force!(F, G)
        a = atom_create(G, s, cell, pbc)
        x = grad_describe(a, desc)
        fill!(F, 0.0)
        for b in 1:nE
            F .+= wE[b] * ref_force(x, xE[b])
        end
        F .*= vE
    end

    optimizer = LBFGS(linesearch=LineSearches.BackTracking())
    sol       = optimize( perturbed_energy, perturbed_force!, G0, optimizer,
                          Optim.Options( g_tol=1e-4, x_abstol=1e-6, iterations=100, show_trace=false, show_every=5, store_trace=true, extended_trace=true ))

    # identify minimum
    gi = [ iter.g_norm  for iter in Optim.trace(sol) ]
    pt = nothing
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
    else
        leap    = [ gi[k+1]/gi[k]   for k in pt ]
        stop    = pt[findfirst( leap.>1 )]
        G_final = Optim.trace(sol)[stop].metadata["x"]
    end

    return G_final
end

#----


function perturb_relax( config, strain ; desc, xE, wE, vE )
    # initial molecular system information
    G, s, cell, pbc = atom_extract(config)

    # forward
    Gp   = relax( cell*(I + strain), G*(I + strain) ; s, pbc, desc, xE, wE, vE )
    plus = atom_create( Gp, s, cell*(I + strain), pbc )

    # backward
    Gm    = relax( cell*(I - strain), G*(I - strain) ; s, pbc, desc, xE, wE, vE )
    minus = atom_create( Gm, s, cell*(I - strain), pbc )

    return [ stress_describe( plus, desc ); stress_describe( minus, desc ) ], [ plus.get_volume(); minus.get_volume() ]
end

#----

function stress_sym( μ, Σ ; skip=9 )
    # off diagonal indices
    inds = split(1:ℓ(μ),skip)
    l    = vcat([ s[[2,3,6]]  for s in inds ]...)
    u    = vcat([ s[[4,7,8]]  for s in inds ]...)

    # symmetrize
    # select unique
    z = vcat([ s[[1,5,9,2,3,6]]  for s in inds ]...)
    return  μ[z], Σ[z,z]
end


#---

function perturb_relax( s, w, xC, K, wE, xE ; desc, xyz, task, v, vE, σ², h,nC, nS=2, z=9, j=1e-10 )

    # apply perturbation
    config = ase.read( xyz[task].file, index=string(s-1) )
    xS, Vm = perturb_relax( config, h*[1 0 0; 0 0 0.5; 0 0.5 0] ; desc, xE, wE, vE )

    # covariance
    Kss    = covariance(   xS,      1:nS,        1:nS       ;  C=stress        )   #      18 x 18
    Ks_s   = covariance(  [xS; xC], nS.+(2:nC),  1:nS       ;  C=stress        )   # 9(nC-1) x 18
    Kes_s  = covariance(  [xS; xC], 1:nS,        nS.+(1:nC) ;  C=stress_energy )'  #      nC x 18
    Ks     = [Kes_s  ; Ks_s]

    # infer energies and covariances
    noise  =  [ σ²[1]*ones(nC);  σ²[2]*ones(ℓ(w)-nC) ]
    KS     =   v*Kss     -   (v*Ks)'*( (v*K + diagm(noise)) \  (v*Ks))   # 18 x 18
    μS     =  (v*Ks)'*w                                                  # 18 x 1

    # remove redundant stress
    μS, KS = stress_sym( μS, KS ; skip=9 )

    # propagate to elastic constant prediction
    M = [ I(6)/(2*h*Vm[1])  -I(6)/(2*h*Vm[2]) ]  # 1 x 12
    μC = M*μS     # 6 x 1
    KC = M*KS*M'  # 6 x 6
    
    # Bulk modulus
    M  = [1 0 2 0 0 0]/3
    μM = M*μC # scalar
    KM = M*KC*M'

    # prediction set properties
    y = xyz[task].EC
    i = xyz[task].Si
    full    = (μC-y)'      *( H(KC      ; j) \ (μC-y) )
    reduced = (μC[i]-y[i])'*( H(KC[i,i] ; j) \ (μC[i]-y[i]) )

    return KS, y-μC, KC, xyz[task].MOD-μM[1], KM[1], full, reduced, i, Vm
end


# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------


function component_wise( μ, σ, y ; α, cal_frac=0.5, nSet=24, bonf=false  )

    # calibration scores
    αb = bonf ? 1 - (1-α)/24 : α
    s = abs.( μ - y ) ./ σ

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum([1., ceil( (ℓ(c)+1)*αb )/ℓ(c)])  )
    
    # coverage
    cwcov = componentwise_coverage( μ[t], q*σ[t], y[t] )
    mvcov = set_coverage( μ[t], q*σ[t], y[t] ; nSet )
    dist  = distance2boundary( μ[t], q*σ[t], y[t], σ[t], nSet )

    return [ [q,α], cwcov, mvcov, dist] 
end

#-------

function predict_componentwise(  qα ; KS, errC, KC, errM, KM, j=1e-8, h, Vm, i, z=6, seed )

    # stress set volume
    q, α = qα
    svol = prod( 2q*Σ2σ(KS) )

    # elastic constant set volume
    rr    =  q * sum( reshape( Σ2σ(KS),z,2) ./ (2*h*Vm)' ; dims=2)
    cvol   = exp( (1/length(rr))    *  sum( log.( 2*rr) ) )  
    cvolr  = exp( (1/length(rr[i])) *  sum( log.( 2*rr[i]) ) )

    # coverage
    ub = errC .<  rr
    lb = errC .> -rr
    full    = prod(ub)*prod(lb)
    red     = prod(ub[i])*prod(lb[i])
    cw_full = sum( ub.*lb )
    cw_red  = sum( ub[i].*lb[i] )

    return [ seed; α; errC; errM; svol; cvol; cvolr; full; red; cw_full; cw_red; 2*q*sqrt(KM); (errM^2/KM)<q^2 ]
end



# ---------------------------------------------------------------
# ===============================================================
#
# Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------


function multivar( μ, Σ, y ; α, cal_frac=0.95 )

    # compute score
    s = [ (yi-μi)'*( H(Σi) \(yi-μi))   for (yi,μi,Σi) in zip(y,μ,Σ) ]

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum([1., ceil( (ℓ(c)+1)*α )/ℓ(c)]) )

    # coverage
    cwcov = componentwise_coverage( μ[t], Σ[t], y[t], q )
    mvcov = set_coverage( μ[t], Σ[t], y[t], q )
    dist  = mean([ relu( (t-m)'*( H(C) \(t-m)) - q ) for (t,m,C) in zip(y[t],μ[t],Σ[t]) ])

    return [[q,α], cwcov, mvcov, dist] 
end

#------

function predict_multivariate( qα ; KS, errC, KC, errM, KM, full, reduced, i, j=1e-8, seed )

    # stress set volume
    q, α = qα
    svol = V_ellipsoid(size(KS,1), q, H(KS;j))

    # elastic constant set volume
    cvol  = exp( (1/size(KC,1))  *  log_V_ellipsoid(size(KC,1), q, H(KC;j)) )
    cvolr = exp( (1/ℓ(i))        *  log_V_ellipsoid(      ℓ(i), q, H(KC[i,i];j)) )

    # vector coverage
    mv_full = full < q
    mv_red  = reduced < q

    # componentwise coverage
    rr     = sqrt(q)*sqrt.(diag(KC))
    comps  = ( errC .< rr ).*( errC.> -rr )
    cw_full_upper = sum(comps)
    cw_red_upper  = sum(comps[i])

    rr     = sqrt(q) ./ sqrt.(diag(inv(KC)))
    comps  = ( errC .< rr ).*( errC.> -rr )
    cw_full = sum(comps)
    cw_red  = sum(comps[i])

    return [ svol; cvol; cvolr; mv_full; mv_red; cw_full; cw_red; cw_full_upper; cw_red_upper; 2*sqrt(q*KM); (errM^2)/KM<q ]
end



# ---------------------------------------------------------------
# ===============================================================
#
# Set up test
#
# ===============================================================
# ---------------------------------------------------------------

#   2 atoms: 225:328
#  16 atoms: 329:548
#  54 atoms: 549:658
# 128 atoms: 659:713

# identify equilibrium configuration for a given size
sz          = length(ARGS) > 0 ? parse(Int64, ARGS[3] ) : 16
equilibrium = Dict( 2  =>225,
                    16 =>329,
                    54 =>549,
                    128=>659 )
config      = equilibrium[sz]

tune_seed    = 1
diamonds     = 225:713
diamonds_16  = 329:548
near_equil   = 329:380
tune_set     = setdiff( diamonds, collect(keys(equilibrium)) )
seed_set     = 2:3#collect(2:111)
files        = [ path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
task         =  "dia"
h            = length(ARGS) > 0 ? parse( Float64, ARGS[4] ) : 0.05
energy_train = length(ARGS) > 0 ? parse( Int64,   ARGS[5] ) : 350
stress_train = length(ARGS) > 0 ? parse( Int64,   ARGS[6] ) : 10
calibrate    = length(ARGS) > 0 ? parse( Int64,   ARGS[7] ) : 100

num_tasks   = length(ARGS) > 0 ? parse(Int64, ARGS[2] ) : 1
loc_id      = length(ARGS) > 0 ? parse(Int64, ARGS[1] ) : 1
loc_seeds   = seed_set[loc_id:num_tasks:end]

# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [[near_equil,diamonds_16,diamonds]], stresses  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )

!isfile( path*"/data/covariance/stress_compiled_K.csv" ) && error("Precomputed stress covariance file is absent.")

for seed in loc_seeds
    elastic_constant_from_stress( config, xyz, settings, loc_id, sz ; task, seed, tune_set, tune_seed, h, core=[stress_train], target=[calibrate], energy_core=[energy_train] )
end


