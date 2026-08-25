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
path = @__DIR__
path  = path[1:end-4]
include( path*"/src/modules/auxiliary.jl")
include( path*"/src/modules/material_settings.jl")
include( path*"/src/modules/quip_descriptors.jl")
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")


# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------

function elastic_constant_from_energy( config, xyz, settings, loc_id, sz; seed, j=1e-8, α_set=collect(0.4:0.05:0.95),
                                                      ε=0.05,h=0.05,
                                                      n_tune=20, fold=20, tune_seed=1, tune_set, 
						                              task="dia", σ²=1e-8, v=1.0, core=[350], target=[100]  )


    # identify tuning configurations to remove from train and calibration sets
    Random.seed!(tune_seed);
    S  = split( sample( tune_set, n_tune, replace=false  ), r(n_tune/fold) )

    # draw configuration indices
    Random.seed!(seed);
    T =      sample( setdiff( xyz[task].inds, [config;S]   ), maximum(target), replace=false )       # test
    C = [1 ; sample( setdiff( xyz[task].inds, [config;T;S] ), maximum(core),   replace=false )   ]   # train
    
    # iterate over training set sizes
    for (nC,nT) in Iterators.product(core,target)
        # train GP
        w, μ, Σ, xC, K, eC, eT, err, ρ = train( task, xyz ; settings, C=C[1:nC], T=T[1:nT], σ², v )
        
        # vanilla conformal scores
        cw  = [ component_wise( μ, Σ2σ(Σ), eT ; α            ) for α in α_set ]
        bf  = [ component_wise( μ, Σ2σ(Σ), eT ; α, bonf=true ) for α in α_set ]

        # multivariate conformal scores
        μ, Σ, eT =   extract(  μ, Σ, eT ; nT, nCal=nT*2 )
        mv       = [ multivar( μ, Σ, eT ; α, j )     for α in α_set ]
        
        # predict elastic constant
        sd   = settings.descriptor
        desc = SOAPDescriptor(  species=sd.species, n_max=sd.n_max, l_max=sd.l_max , sigma=sd.sigma, r_cut=sd.r_cut )
        
        # predict energies at perturbed geometries
        KE, errC, KC, errM, KM, full, reduced, i, Vm  = perturb_relax( config, w, xC, K ; desc, xyz, task, v, σ², h, ε, nC )

        # apply conformal
        cw_out = predict_componentwise.( first.(cw) ; KE, errC, KC, errM, KM, i, Vm, h, ε, seed   )
        bf_out = predict_componentwise.( first.(bf) ; KE, errC, KC, errM, KM, i, Vm, h, ε, seed   )
        mv_out = predict_multivariate.(  first.(mv) ; KE, errC, KC, errM, KM, i, j, full, reduced, seed  )
	   
        # save results
        report( loc_id, sz, cw_out, mv_out, bf_out, nC, nT, config, ε, h )
    end
end


# ---------------------------------------------------------------
# ===============================================================
#
# Report
#
# ===============================================================
# ---------------------------------------------------------------


function report( loc_id, sz, cw, mv, bf, nC, nT, config, ε, h )
    path_cf = string( path, "/data/energy_elastic_constant/ec_energy_config", sz, "_", ε, "_", h, "_", "nC", nC, "_nT", nT, "_", loc_id, "_conformal.csv" )
    h_cf    = [ :seed, :cw_alpha, :ec1, :ec2, :ec3, :ec4, :ec5, :ec6, :bm,
                :cw_energy_volume, :cw_ec_volume, :cw_ec_volume_red,  :cw_fullcov, :cw_redcov, :cw_cwfullcov, :cw_cwredcov, :cw_Mvol, :cw_Mcov,
                :mv_energy_volume, :mv_ec_volume, :mv_ec_volume_red,  :mv_fullcov, :mv_redcov, :mv_cwfullcov, :mv_cwredcov, :mv_cwfullcov_upper, 
                :mv_cwredcov_upper, :mv_Mvol, :mv_Mcov  ]
    addToFile( DataFrame( [hcat(cw...)'  hcat(mv...)'], h_cf ), path_cf )


    path_bf = string( path, "/data/energy_elastic_constant/ec_energy_config", sz, "_", ε, "_", h, "_", "nC", nC, "_nT", nT, "_", loc_id, "_bonf.csv" )
    h_bf    = [ :seed, :cw_alpha, :ec1, :ec2, :ec3, :ec4, :ec5, :ec6, :bm,
               :cw_energy_volume, :cw_ec_volume, :cw_ec_volume_red,  :cw_fullcov, :cw_redcov, :cw_cwfullcov, :cw_cwredcov, :cw_Mvol, :cw_Mcov ]
    addToFile( DataFrame( hcat(bf...)', h_bf ), path_bf )
end



# ---------------------------------------------------------------
# ===============================================================
#
# Training
#
# ===============================================================
# ---------------------------------------------------------------

function train( task, xyz ; settings, C, T, σ²=0.001, v=10.0  )

    # construct covariance structures
    xC, K, Ks, Kss, eC, eT = covariance( task, xyz ; settings, C, T)

    # train GP 
    w = (v*H(K) + σ²*I) \ eC
    μ = (v*Ks)'*w
    Σ = v*Kss - (v*Ks)'*( (v*H(K) + σ²*I) \  (v*Ks) )

    # report error
    err = rmse(μ,eT)
    ρ   =  cor(μ,eT)

    return w, μ, Σ, xC, K, eC, eT, err, ρ
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
    Ke_ss  = covariance(  [xC; xT], nC.+(1:nT), nC.+(1:nT) ; C=energy )

    return xC, Ke, Ke_s, Ke_ss, vcat(eC...), vcat(eT...)
end

#----------

function extract(m, C, y; nConfig=24, nT, nCal)
    iCal = [ sample(1:nT, nConfig) for _ in 1:nCal ]
    return [m[i] for i in iCal], [C[i,i] for i in iCal  ], [y[i] for i in iCal]
end

# ---------------------------------------------------------------
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

function relax(cell, G0; s, pbc, desc, xC, w, v)
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

    optimizer = LBFGS(linesearch=LineSearches.BackTracking())
    sol       = optimize( perturbed_energy, perturbed_force!, G0, optimizer,
                          Optim.Options( g_tol=1e-4, x_abstol=1e-6, iterations=100, show_trace=false, store_trace=true, extended_trace=true ))

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

function perturb_relax( config, u, strain ; desc, xC, w , v )

    # initial molecular system information
    G, s, cell, pbc = atom_extract(config)

    # Compute volume at positive and negative perturbations
    Vp = atom_create( G, s, cell*(I + strain), pbc ).get_volume()
    Vm = atom_create( G, s, cell*(I - strain), pbc ).get_volume()

    X = []
    for (b,a) in Iterators.product([+,-],[+,-])
        c  = cell*a(I, strain)*b(I, u)
        g0 =    G*a(I, strain)*b(I, u)

        G1 = relax(c, g0 ; s, pbc, desc, xC, w, v )
        a  = atom_create( G1, s, c, pbc )

        push!( X, describe( a, desc ))
    end
    return (X, [Vp;Vp;Vm;Vm])
end

#----

function perturb_no_shear( config, u, strain ; desc, xC, w , v )

    # initial molecular system information
    G, s, cell, pbc = atom_extract(config)

    # Compute volume at positive and negative perturbations
    Vp = atom_create( G, s, cell*(I + strain), pbc ).get_volume()
    Vm = atom_create( G, s, cell*(I - strain), pbc ).get_volume()

    X = []
    for (b,a) in Iterators.product([+,-],[+,-])
        c  = cell*a(I, strain)*b(I, u)
        g0 =    G*a(I, strain)*b(I, u)

        a  = atom_create( g0, s, c, pbc )

        push!( X, describe( a, desc ))
    end
    return (X, [Vp;Vp;Vm;Vm])
end


#----

function perturb_relax( s, w, xC, K ; desc, xyz, task, v, σ², h, ε, nC, nS=24, j=1e-10  )

    # apply perturbation
    strain = h*[1 0 0; 0 0 0.5; 0 0.5 0]
    config = ase.read( xyz[task].file, index=string(s-1) )
    #diags  =      [  perturb_relax(   config, ε*diagm(ej(i)),                      strain ; desc, xC, w, v )  for i in   1:3]
    diags  =      [ perturb_no_shear( config, ε*diagm(ej(i)),                      strain ; desc, xC, w, v )  for i in   1:3]
    off    = vcat([[ perturb_relax(   config, 0.5*ε*(ej(i)*ej(j)' + ej(j)*ej(i)'), strain ; desc, xC, w, v )  for i in j+1:3] for j in 1:2 ]...)
    
    # extract volume and relaxed geometry
    Vm     = vcat( last.([diags;off])...)
    xS     = vcat(first.([diags;off])...)

    # predict energies and elastic constants
    Ks  = covariance(  [xS; xC], 1:nS, nS.+(1:nC) ; C=energy )
    Kss = covariance(   xS,      1:nS,      1:nS  ; C=energy )

    # infer energies and covariances
    μE = v*Ks * w
    KE = v*Kss - (v*Ks)*( (v*K + σ²*I) \  (v*Ks') )

    # propagate to elastic constant prediction
    M  = elastic_constant_operator(ε,h,Vm)
    μC = M*μE
    KC = M*KE*M'

    # bulk modulus
    M  = [1 0 2 0 0 0]/3
    μM = M*μC # scalar
    KM = M*KC*M'
    KM = maximum([KM[1],1e-16])

    # scores
    y = xyz[task].EC
    i = xyz[task].Si
    full    = (μC-y)'      *( H(KC      ; j) \ (μC-y) )
    reduced = (μC[i]-y[i])'*( H(KC[i,i] ; j) \ (μC[i]-y[i]) )

    return KE, y-μC, KC, xyz[task].MOD-μM[1], KM[1], full, reduced, i, Vm
end




# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------


function component_wise( μ, σ, y ; α, cal_frac=0.95, nSet=24, bonf=false  )

    # bonferroni
    αb = bonf ? 1 - (1-α)/24 : α

    # calibration scores
    s = abs.( μ - y ) ./ σ

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum( [ 1., ceil( (ℓ(c)+1)*αb )/ℓ(c) ] ) )

    # coverage
    cwcov = componentwise_coverage( μ[t], q*σ[t], y[t] )
    mvcov = set_coverage( μ[t], q*σ[t], y[t] ; nSet )
    dist  = distance2boundary( μ[t], q*σ[t], y[t], σ[t], nSet )

    return [ [q,α], cwcov, mvcov, dist] 
end

#-------

function predict_componentwise(  qα ; KE, errC, KC, errM, KM, ε, h, Vm, i, seed )

    # stress set volume
    q, α = qα
    evol = prod( 2q*Σ2σ(KE) )

    # elastic constant set volume
    rr    =  q * reshape(sum( reshape( Σ2σ(KE),4,6) ./ (4ε*h*reshape(Vm,4,6))   ; dims=1 ),:,1)
    cvol   = exp( (1/length(rr))    *  sum( log.( 2*rr) ) )
    cvolr  = exp( (1/length(rr[i])) *  sum( log.( 2*rr[i]) ) )

    # coverage
    ub = errC .<  rr
    lb = errC .> -rr
    full    = prod(ub)*prod(lb)
    red     = prod(ub[i])*prod(lb[i])
    cw_full = sum( ub.*lb )
    cw_red  = sum( ub[i].*lb[i] )

    return [ seed; α; errC; errM; evol; cvol; cvolr; full; red; cw_full; cw_red; 2*q*sqrt(KM); (errM^2/KM)<q^2 ]
end



# ---------------------------------------------------------------
# ===============================================================
#
# Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------


function multivar( μ, Σ, y ; α, cal_frac=0.95, j )

    # compute score
    s = [ (yi-μi)'*( H(Σi;j) \(yi-μi))   for (yi,μi,Σi) in zip(y,μ,Σ) ]
    
    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum([1., ceil( (ℓ(c)+1)*α )/ℓ(c)]) )

    # coverage
    cwcov = componentwise_coverage( μ[t], H.(Σ[t];j), y[t], q )
    mvcov = set_coverage( μ[t], Σ[t], y[t], q )
    dist  = mean([ relu( (t-m)'*( H(C;j) \(t-m)) - q ) for (t,m,C) in zip(y[t],μ[t],Σ[t]) ])

    return [[q,α], cwcov, mvcov, dist] 
end

#------

function predict_multivariate( qα ; KE, errC, KC, errM, KM, full, reduced, i, j, seed )

    # stress set volume
    q, α = qα
    evol = V_ellipsoid(size(KE,1), q, H(KE;j))

    # elastic constant set volume
    cvol  = exp( (1/size(KC,1))  *  log_V_ellipsoid(size(KC,1), q, H(KC;j)) )
    cvolr = exp( (1/ℓ(i))        *  log_V_ellipsoid(      ℓ(i), q, H(KC[i,i];j)) )

    # vector coverage
    mv_full = full < q
    mv_red  = reduced < q

    # componentwise coverage
    md      = minimum(diag(H(KC;j)))
    KC      = md < 0. ? H(KC;j) +  1.1*abs(md)*I : H(KC;j)
 
    rr      = q ./ diag(inv(H(KC)))			   
    comps   = errC.^2 .< rr 
    cw_full = sum(comps)
    cw_red  = sum(comps[i])

    rr      = q .* diag(KC)
    comps   = errC.^2 .< rr
    cw_full_upper = sum(comps)
    cw_red_upper  = sum(comps[i])

    return [  evol; cvol; cvolr; mv_full; mv_red; cw_full; cw_red; cw_full_upper; cw_red_upper;  2*sqrt(q*KM); (errM^2/KM)<q ]
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
tune_set     = setdiff( diamonds, collect(keys(equilibrium)) )
seed_set     = collect(2:3)#collect(2:211)
files        = [ path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
task         =  "dia"
ε            = length(ARGS) > 0 ? parse( Float64, ARGS[4] ) : 0.05
h            = length(ARGS) > 0 ? parse( Float64, ARGS[5] ) : 0.05
train_n      = length(ARGS) > 0 ? parse( Int64,   ARGS[6] ) : 350
calibrate_n  = length(ARGS) > 0 ? parse( Int64,   ARGS[7] ) : 100

num_tasks   = length(ARGS) > 0 ? parse(Int64, ARGS[2] ) : 1
loc_id      = length(ARGS) > 0 ? parse(Int64, ARGS[1] ) : 1
loc_seeds   = seed_set[loc_id:num_tasks:end]

# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [diamonds], stresses  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )

for seed in loc_seeds
    elastic_constant_from_energy( config, xyz, settings, loc_id, sz ; task, seed, tune_set, tune_seed, ε, h, core=[train_n], target=[calibrate_n] )
end
