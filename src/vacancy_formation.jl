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
using Clustering

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


# use of 0.9 does not lead to improvement
function vacancy_formation( S, tasks, xyz, settings; path, j=1e-8, seed, 
						     n_tune=(p=20,s=10), fold=(p=20,s=10), tune_seed=1, tune_set, 
						     α_set=collect(0.4:0.05:0.95), σ²=(p=1e-4,s=1e-4), v=(p=1,s=1,ratio=10000), ρ=0.55,
                             core=(p=[350],s=[150]), target=(p=[100],s=[40]), scale=10000  )

    # draw configuration indices
    Random.seed!(tune_seed);
    Hp = sample( tune_set.p, n_tune.p, replace=false  ), r(n_tune.p/fold.p) 
    Hs = sample( tune_set.s, n_tune.s, replace=false  ), r(n_tune.s/fold.s) 

    # draw configuration indices
    Random.seed!(seed);
    C = ( p=[1 ; sample( setdiff( xyz[tasks.p].inds, [S;Hp]          ), maximum(core.p)-1, replace=false  )],  # train
          s=     sample( setdiff( xyz[tasks.s].inds, [S;Hs]          ), maximum(core.s),   replace=false  ))
    T = ( p=     sample( setdiff( xyz[tasks.p].inds, [S;C.p;Hp]     ), maximum(target.p), replace=false  ),  # test
          s=     sample( setdiff( xyz[tasks.s].inds, [S;C.s;Hs]     ), maximum(target.s), replace=false  ))


    # iterate over training set sizes
    for (nCp,nCs,nTp,nTs) in Iterators.product(core.p,core.s,target.p,target.s)

        w, μ, Σ, xCp, xCs, K, Ks, eTp, eTs, η, err, pear, yT = train( tasks ; C=(p=C.p[1:nCp] , s=C.s[1:nCs]), T=(p=T.p[1:nTp] , s=T.s[1:nTs]), σ², v, ρ  )

        # univariate conformal
        cw  = [ component_wise( μ, Σ2σ(Σ), eTp, eTs ; α            ) for α in α_set ]
        cb  = [ component_wise( μ, Σ2σ(Σ), eTp, eTs ; α, bonf=true ) for α in α_set ]

        # multivariate conformal scores
        μs, Σ, eT = extract(  μ, Σ,  eTp, eTs ; nCal=length(T.p)*2 )
        
        # config size
        sd   = settings.descriptor
        desc = SOAPDescriptor(  species=sd.species, n_max=sd.n_max, l_max=sd.l_max , sigma=sd.sigma, r_cut=sd.r_cut )
        M    = size( atom_extract( ase.read( xyz[tasks.p].file, index=string(S-1) ) )[1], 1)
        
        # multivariate conformal
        mv  =  [ multivar(   μs, Σ, eT ; α, j      ) for α in α_set ]
        mM  =  [ M_multivar( μs, Σ, eT ; α, j, n=M ) for α in α_set ]

        # predict vacancy energy
        KE, errV, KV, n  = remove_relax( S, w, xCp, xCs, K ; desc, xyz, tasks, v, σ², η, ρ,  reset, scale )

        # apply conformal
        cw_out = predict_componentwise.(  first.(cw) ; KE, errV, KV, n        )
        cb_out = predict_componentwise.(  first.(cb) ; KE, errV, KV, n        )
        mv_out = predict_multivariate.(   first.(mv) ; KE, errV, KV, j, seed  )
        mm_out = predict_M_multivariate.( first.(mM) ; KE, errV, KV, j, seed  )
        report( S, cw_out, cb_out, mv_out, mm_out, nCp, nCs, nTp, nTs, path  )
    end
end

#------------------

function extract(m, C, yp, ys; nConfig=2, nCal)
    ip   = 1:length(yp)
    is   = length(yp).+(1:length(ys))
    iCal = [  [sample(ip),sample(is)]  for _ in 1:nCal ]
    return [m[i] for i in iCal], [C[i,i] for i in iCal  ], [ [yp;ys][i] for i in iCal]
end


# ---------------------------------------------------------------
# ===============================================================
#
# Report
#
# ===============================================================
# ---------------------------------------------------------------

function report( S, cw, cb, mv, mm, nCp, nCs, nTp, nTs, path )
    path_cf = string( path, "_",S, "_", nCp, "_", nCs, "_", nTp, "_", nTs,  "_conformal.csv" )
    h_cf    = [ :seed, :alpha, :VE, :mv_energy_volume, :mv_vfe_volume, :mv_vfe_cov,         
                                             :mm_energy_volume, :mm_vfe_volume, :mm_vfe_cov,
                                             :cb_energy_volume, :cb_vfe_volume, :cb_vfe_cov,
                                             :cw_energy_volume, :cw_vfe_volume, :cw_vfe_cov  ]
    addToFile( DataFrame( [hcat(mv...)' hcat(mm...)' hcat(cb...)'  hcat(cw...)'], h_cf ), path_cf )    
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

function train( tasks ; C, T, σ², v, ρ  )

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
    Kss      = covariance(  [xCp;xCs;xTp;xTs], nC.+(1:nT),  nC.+(1:nT) ; C=energy )

    # symmetrize
    K = (K+K')/2

    # separate indices
    nCp, nCs, nTp, nTs = ℓ(C.p), ℓ(C.s), ℓ(T.p), ℓ(T.s)

    # construct multitask training covariance
    nCp, nCs, nTp, nTs               = ℓ(C.p), ℓ(C.s), ℓ(T.p), ℓ(T.s)
    K[ 1:nCp,        1:nCp        ] *= v.p  
    K[ 1:nCp,        nCp.+(1:nCs) ] *= v.p*ρ
    K[ nCp.+(1:nCs), 1:nCp        ] *= v.p*ρ
    K[ nCp.+(1:nCs), nCp.+(1:nCs) ] *= v.p*ρ^2 + v.s 
    
    # construct noise vector
    η = [ repeat( [σ².p], nCp ); repeat( [σ².s], nCs ) ] ./ v.ratio

    # construct multitask train-test covariance
    Ks[ 1:nCp,        1:nTp        ] *= v.p  
    Ks[ 1:nCp,        nTp.+(1:nTs) ] *= v.p*ρ
    Ks[ nCp.+(1:nCs), 1:nTp        ] *= v.p*ρ
    Ks[ nCp.+(1:nCs), nTp.+(1:nTs) ] *= v.p*ρ^2 + v.s 

    # construct multitask test covariance
    Kss[ 1:nTp,        1:nTp        ] *= v.p
    Kss[ 1:nTp,        nTp.+(1:nTs) ] *= v.p*ρ
    Kss[ nTp.+(1:nTs), 1:nTp        ] *= v.p*ρ
    Kss[ nTp.+(1:nTs), nTp.+(1:nTs) ] *= v.p*ρ^2 + v.s

    # test
    w = ( K + diagm(η) ) \ [eCp;eCs]
    μ = Ks'*w
    Σ = v.ratio * ( Kss - Ks'*( ( K + diagm(η) )  \Ks) )

    # evaluate
    err  = [ rmse( μ[1:nTp]./nap, eTp./nap), rmse( μ[nTp.+(1:nTs)]./nas, eTs./nas) ]
    pear = [ cor(  μ[1:nTp], eTp), cor(  μ[nTp.+(1:nTs)], eTs) ]

    return w, μ, Σ, xCp, xCs, K, Ks, eTp, eTs, η, err, pear, [eCp;eCs]
end

#-------

covariance( X, aa, bb ; C ) = vcat([ hcat([ C(X[a], X[b])  for b in bb ]...)  for a in aa ]...)
energy( A, B ;  ζ=4  )      = sum((A*B').^ζ)



# ---------------------------------------------------------------
# ===============================================================
#
# Relax geometry
#
# ===============================================================
# ---------------------------------------------------------------

function ref_energy(A, B; ζ=4)
    return sum((A * B') .^ ζ)
end

#-----

function ref_force(A, B; ζ=4)
    δ = ζ * (last(A) * B') .^ (ζ - 1)
    @tullio fe[i, j] := δ[a, b] * B[b, r] * first(A)[a, i, j, r]
    return fe
end

#-----

function relax(cell, G0; s, pbc, desc, xCp, xCs, w, v, ρ)
    nCp = length(xCp)
    nCs = length(xCs)

   # return G0
    function perturbed_energy(G)
        a  = atom_create(G, s, cell, pbc)
        x  = describe(a, desc)
        E  =  ρ  *v.p        * sum(w[b]     * ref_energy(x, xCp[b]) for b in 1:nCp)
        E += (ρ^2*v.p + v.s) * sum(w[b+nCp] * ref_energy(x, xCs[b]) for b in 1:nCs)
        return E
    end

    function perturbed_force!(F, G)
        a = atom_create(G, s, cell, pbc)
        x = grad_describe(a, desc)
        fill!(F, 0.0)
        F .+=  ρ  *v.p        * sum(w[b]     * ref_force(x, xCp[b]) for b in 1:nCp)
        F .+= (ρ^2*v.p + v.s) * sum(w[b+nCp] * ref_force(x, xCs[b]) for b in 1:nCs)
    end

    optimizer = LBFGS(linesearch=LineSearches.BackTracking())
    sol       = optimize( perturbed_energy, perturbed_force!, G0, optimizer,
                          Optim.Options( g_tol=1e-1, x_abstol=1e-6, iterations=100, show_trace=false, extended_trace=true, store_trace=true )) 

    gi = [ iter.g_norm  for iter in Optim.trace(sol) ]
    pt = nothing
    for j in 2:length(gi)
        for k in 1:j-1
            if gi[j]/gi[k] > 5
                pt=k
            end
        end
        if !isnothing(pt)
            break
        end
    end

    if isnothing(pt)
        G_final = Optim.minimizer(sol)
    else
        gr      = gi[2:pt] ./ gi[1:pt-1]
        stop    = findlast( gr.<1 )
        G_final = Optim.trace(sol)[stop+1].metadata["x"]
    end
    return G_final
end

function remove_relax( S, w, xCp, xCs, K ; desc, xyz, tasks, v, σ², ρ, η,  reset, scale, truth=3.67 )

    # read full configuration
    config = ase.read( xyz[tasks.p].file, index=string(S-1) )

    # extract geometry
    G, s, cell, pbc = atom_extract(config)

    # remove atom
    svac   = string("Si",size(G,1)-1)
    G1     = relax(cell, G[1:end-1,:] ; s=svac, pbc, desc, xCp, xCs, w, v, ρ )
    vacant = atom_create( G1, svac, cell, pbc )

    # compute descriptors 
    X = [describe( config, desc ), describe( vacant, desc )]

    # predict energies and elastic constants
    nC  = length(xCp)+length(xCs)
    Ks  = covariance(  [xCp;xCs;X], 1:nC, nC.+(1:2) ; C=energy );
    Kss = covariance(           X , 1:2,  1:2,      ; C=energy );

    # adjust to create multitask covariance
    nCp = length(xCp)
    nCs = length(xCs)
    Ks[      1:nCp, 1]  *=      v.p;
    Ks[      1:nCp, 2]  *=  ρ  *v.p;
    Ks[nCp.+(1:nCs),1]  *=  ρ  *v.p;
    Ks[nCp.+(1:nCs),2]  *= (ρ^2*v.p + v.s);
    Kss[1,1] *=      v.p;
    Kss[1,2] *=  ρ  *v.p;
    Kss[2,1] *=  ρ  *v.p;
    Kss[2,2] *= (ρ^2*v.p + v.s);

    # infer energies and covariances
    μE    = Ks' * w
    KE    = v.ratio*( Kss - Ks'*( (K + diagm(η))  \Ks) )

    # predict vancancy energy
    n  = size(G,1)
    M  = [ -(n-1)/n  1 ]
    μV = M*μE
    KV = M*KE*M'

    return KE, truth - μV[1], KV[1], n
end


function simsigma(K, Ks ; η, induce, scale, reset)
    
    M = (K[induce,:]/scale)*(K[:,induce]/scale)  +  sqrt.(η[induce]).*(K[induce,induce]/scale^2).*sqrt.(η[induce]')
    C = Ks[induce,:]' *( ( (M+M')/2 + (1e-8)*I  ) \ (Ks[induce,:])  )
    
    return (C+C')/(2*reset)
end


# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------

function component_wise( μ, σ, eTp, eTs ; α, cal_frac=0.95, nSet=2, bonf=false  )

    αα = bonf ? 1 - (1-α)/nSet : α

    # calibration scores
    ip = 1:length(eTp)
    is = length(eTp).+(1:length(eTs))
    sp = abs.( μ[ip] - eTp ) ./ σ[ip]
    ss = abs.( μ[is] - eTs ) ./ σ[is]

    # quantile
    cp = 1:r(ℓ(sp)*cal_frac)
    cs = 1:r(ℓ(ss)*cal_frac)
    tp = setdiff(1:ℓ(sp),cp)
    ts = setdiff(1:ℓ(ss),cs)
    qp = quantile( sp[cp], minimum( [1, ceil( (ℓ(cp)+1)*αα )/ℓ(cp) ]) )
    qs = quantile( ss[cs], minimum( [1, ceil( (ℓ(cs)+1)*αα )/ℓ(cs) ]) )



    # coverage
    cwcov_p = componentwise_coverage( μ[ip][tp], qp*σ[ip][tp], eTp[tp] )
    cwcov_s = componentwise_coverage( μ[is][ts], qs*σ[is][ts], eTs[ts] )

    mvcov   = paired_set_coverage( μ[ip][tp], μ[is][ts], qp*σ[ip][tp], qs*σ[is][ts], eTp[tp], eTs[ts] )

    return [ [qp,qs,αα], cwcov_p, cwcov_s, mvcov]
end
function predict_componentwise(  qα ; KE, errV, KV, n)

    # stress set volume
    qp, qs, α = qα
    evol =  4*qp*qs*sqrt(KE[1,1]*KE[2,2])

    # elastic constant set volume
    rr   = qp*sqrt(KE[1,1])*(n-1)/n + qs*sqrt(KE[2,2])
    vvol = 2*rr

    # coverage
    vcov = abs(errV) < rr

    return [ evol; vvol; vcov ]
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
    s = [ (yi-μi)'*( H(Σi;j) \ (yi-μi))   for (yi,μi,Σi) in zip(y,μ,Σ) ]
    
    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum( [ ceil( (ℓ(c)+1)*α )/ℓ(c) ] ))


    # coverage
    cwcov = componentwise_coverage( μ[t], Σ[t], y[t], q )
    mvcov = set_coverage( μ[t], Σ[t], y[t], q )

    return [[q,α], cwcov, mvcov]
end

function predict_multivariate( qα ; KE, errV, KV, j, seed )

    # stress set volume
    q, α = qα
    evol = V_ellipsoid(size(KE,1), q, H(KE;j))

    # elastic constant set volume
    rr   = sqrt(q*KV)
    vvol = 2*rr

    # coverage
    vcov = abs(errV) < rr

    return [ seed; α; errV; evol; vvol; vcov  ]
end

function M_multivar( μ, Σ, y ; α, cal_frac=0.95, j, n )

    # compute score
    M = [ -(n-1)/n ; 1 ]
    s = [ dot(M, (yi-μi))^2 / dot( M, H(Σi;j)*M )   for (yi,μi,Σi) in zip(y,μ,Σ) ]

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], ceil( (ℓ(c)+1)*α )/ℓ(c) )

    # coverage
    cwcov = componentwise_coverage( μ[t], Σ[t], y[t], q )
    mvcov = set_coverage( μ[t], Σ[t], y[t], q )

    return [[q,α], cwcov, mvcov]
end

function predict_M_multivariate( qα ; KE, errV, KV, j, seed )

    # stress set volume
    q, α = qα
    evol = V_ellipsoid(size(KE,1), q, H(KE;j))

    # elastic constant set volume
    rr   = sqrt(q*KV)
    vvol = 2*rr

    # coverage
    vcov = abs(errV) < rr

    return [ evol; vvol; vcov  ]
end



# ---------------------------------------------------------------
# ===============================================================
#
# User defined settings 
#
# ===============================================================
# ---------------------------------------------------------------

#   2 atoms: 225:328
#  16 atoms: 329:548
#  54 atoms: 549:658
# 128 atoms: 659:713
#  63 atoms vacancy: 1876:1975
# 215 atoms vacancy: 1976:2086
# divacancy: 1798:1875


# identify equilibrium configuration for a given size
equilibrium = Dict( 2  =>225,
                    16 =>329,
                    54 =>549,
                    128=>659,
                     63=>1876, 
                    215=>1798 )

tune_seed    = 1
diamonds     = 225:713
vacancies    = 1876:2086
tune_set     = ( p=setdiff( diamonds,  collect(keys(equilibrium)) ),
                 s=setdiff( vacancies, collect(keys(equilibrium)) ) )
seed_set     = [12,13]#collect(12:26)
files        = [ path*"/data/configurations/GAP_silicon.xyz", path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy","dft_energy"]
derivatives  = ["dft_force","dft_force"]
stresses     = ["dft_virial","dft_virial"]
tasks        = ["dia","vacancy"]
path         = path*"/data/vacancy_formation/vacancy_"

# parse arguments
println(ARGS)
n_systems   = length(ARGS) > 0 ? parse(Int64, ARGS[8] ) : 1
config_set  = [ equilibrium[54].+(1:n_systems) ; equilibrium[128].+(1:n_systems) ] .- 1
num_tasks   = length(ARGS) > 0 ? parse(Int64, ARGS[2] ) : 1
loc_id      = length(ARGS) > 0 ? parse(Int64, ARGS[1] ) : 1
ρ           = parse(Float64, ARGS[3] )
core        = ( p=[parse(Int64, ARGS[4])],  s=[parse(Int64, ARGS[5])] )
target      = ( p=[parse(Int64, ARGS[6])],  s=[parse(Int64, ARGS[7])] )
loc         = reshape(collect(Iterators.product(config_set,seed_set)),:,1)[loc_id:num_tasks:end]

# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [diamonds,vacancies], stresses ; tasks  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )
tasks        = (p=tasks[1], s=tasks[2])


for (config,seed) in loc
    vacancy_formation( config, tasks, xyz, settings ; seed, tune_set, tune_seed, path, core, target, ρ )
end

