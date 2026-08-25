using LinearAlgebra
using Random
using CSV
using DataFrames
using Distributions
using Tullio
using SpecialFunctions

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

# import local files
path  = @__DIR__
path  = path[1:end-4]
include(path*"/src/modules/auxiliary.jl")
include(path*"/src/modules/material_settings.jl")
include(path*"/src/modules/quip_descriptors.jl")
include(path*"/src/modules/inner_kernel.jl" )
include(path*"/src/modules/ellipsoid.jl")
include(path*"/src/modules/hypercube.jl")


# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------


function predict( xyz, settings; path, readpath, seed, task="dia", B_quantiles=[0.8,0.9,0.95,0.99], α_set=collect(0.4:0.05:0.95), rc_set=collect(0.01:0.05:0.5), j=1e-4, σ²=1e-3, v=1e5, nT=200 )

    # draw configuration indices
    Random.seed!(seed);
    C = CSV.read( readpath*"C.csv", DataFrame)[:,1]  
    T = sample( setdiff( xyz[task].inds, C ), nT, replace=false  )  

    # read covariance
    s    = CSV.read( readpath*"sor.csv", DataFrame)[:,1]
    name = string( readpath, "SOR_", length(s), "_", length(C) )
    M    = stabilize( Matrix(CSV.read( name*".csv",   DataFrame)) ; j )
    w    = CSV.read( name*"_w.csv", DataFrame)[:,1] 

    # prediction, one configuration at a time
    nS = ℓ(s)
    xC = describe(  task, C[s] ; xyz, settings, type="force" )[1]
    εs, smv, scw, Σs = [], [], [], []
    for (a,t) in enumerate(T)
        
        xT, eT, fT = describe(  task, [t] ; xyz, settings, type="force"    )          
        Ke  = covariance( [xC; xT], 1:nS,       nS.+(1:1) ; C=energy       )
        Kef = covariance( [xC; xT], nS.+(1:1),  1:nS      ; C=force_energy )'
        Kfe = covariance( [xC; xT], 2:nS,       nS.+(1:1) ; C=force_energy )
        Kf  = covariance( [xC; xT], 2:nS,       nS.+(1:1) ; C=force        )

        μ = v * [ Ke Kef ; Kfe Kf ]' * w

        Σ = stabilize( σ² * ( v*[ Ke Kef ; Kfe Kf ])' * ( M \ (v*[ Ke Kef ; Kfe Kf ]) ) ; j )
        y = [ eT[1] ; -fT[1] ]

        diff = (μ - y)
        push!( εs, diff )
        push!( smv, dot( diff, Σ\diff)  )
        push!( scw, abs.(diff)./ Σ2σ(Σ) )
        push!( Σs, Σ )

    end
    
    # conformal
    mv  = [  multivar(       smv, Σs, εs ; α )                 for α in α_set  ]
    cw  = [  component_wise( scw, Σs     ; α )                 for α in α_set  ]
    crq = [[ riskcon_robust_quantile( smv, Σs, εs ; d, q, j )  for d in rc_set ]   for q in B_quantiles ]
    cr1 = [[ riskcon_robust_1norm(    smv, Σs, εs ; d, q )     for d in rc_set ]   for q in B_quantiles ]

    report( cw, mv, vcat(cr1...), vcat(crq...), sqrt.(mean(vcat(εs...).^2)), path )
    
end



# ---------------------------------------------------------------
# ===============================================================
#
# Report
#
# ===============================================================
# ---------------------------------------------------------------

function report( cw, mv, rc1, rcq, err, path )

    path_cw  = string( path, "cw_scaled.csv" )
    h_cw     = [:cw,  :mv,  :vol_mean,  :vol_median,  :vol_max]
    addToFile( DataFrame( hcat(cw...)', h_cw ), path_cw )

    path_mv  = string( path, "mv_scaled.csv" )
    h_mv     = [:alpha,  :cw_lo,  :cw_hi,  :mv,  :vol_mean,  :vol_median,  :vol_max]
    addToFile( DataFrame( hcat(mv...)', h_mv ), path_mv )

    path_rc1 = string( path, "cr1_scaled.csv" )
    h_rc     = [:alpha,  :B,  :lambda,  :cw_lo,  :cw_hi,  :mv,  :vol_mean,  :vol_median,  :vol_max,  :guarantee]
    addToFile( DataFrame( hcat(rc1...)', h_rc ), path_rc1 )

    path_rcq = string( path, "crq_scaled.csv" )
    addToFile( DataFrame( hcat(rcq...)', h_rc ), path_rcq )
end


# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------


function component_wise( s, Σs ; α, cal_frac=0.6  )

    # rescale for system size
    lg = size.(Σs,1)
    Σs = copy(Σs) .* lg
    s  = copy(s) ./ sqrt.(lg)

    # calibrate and test
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)

    # split scores
    energies =  first.(s)
    forces   = [ ss[2:end]  for ss in s ]
    
    # quantiles
    ne = length(c)
    nf = sum(length.(forces[c]))
    qe = quantile( energies[c], minimum([1., ceil( (ne+1)*α )/ne]) )
    qf = quantile( vcat(forces[c]...),   minimum([1, ceil( (nf+1)*α )/nf]) )

    # coverage
    cwcov =  mean([ energies[t] .< qe ; vcat(forces[t]...).<qf ]) 
    mvcov =  mean([ prod([e<qe; f.<qf])  for (e,f) in zip(energies[t],forces[t]) ])

    # subset volume
    v = [ exp( (1/length(ss)) * sum( log.(2*qf*ss) ) )  for ss in Σ2σ.(Σs[t]) ]

    return [ cwcov, mvcov, mean(v), median(v), maximum(v) ]
end

# ---------------------------------------------------------------
# ===============================================================
#
# Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------

function stabilize(M;j=1e-8)
    M    = (M+M')/2
    md   = minimum(diag(H(M;j)))
    M    = md < 0. ? H(M;j) +  1.1*abs(md)*I : H(M;j)
    return M 
end

function multivar( s, Σs, εs ; α, cal_frac=0.6, j=1e-8 )

    # scale for different system sizes
    lg = Base.size.(Σs,1)
    s  = copy(s)  ./ lg
    Σs = copy(Σs) .* lg

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum( [1., ceil( (ℓ(c)+1)*α )/ℓ(c)] ) )

    # coverage
    cwcov_l = err_componentwise_coverage( εs[t], Σs[t], q  )
    cwcov_u = err_componentwise_coverage_upper( εs[t], Σs[t], q  )
    mvcov   = mean( s[t] .< q )

    # volume 
    v = [ exp(  (1/Base.size(ss,1))*log_V_ellipsoid(Base.size(ss,1), q, H(ss ;j))  )   for ss in Σs[t] ]

    return [ α, cwcov_l, cwcov_u, mvcov, mean(v), median(v), maximum(v) ] 
end



# ---------------------------------------------------------------
# ===============================================================
#
# Conformal risk control
#
# ===============================================================
# ---------------------------------------------------------------

function riskcon_robust_1norm( s_orig, Σs, εs ; d, q, cal_frac=0.6, j=1e-8 )
    # threshold score
    lg = Base.size.(Σs,1)
    Σs = copy(Σs) .* lg
    s  = copy(s_orig) ./ lg
    n  = r(ℓ(s)*cal_frac)
    c  = 1:n
    t  = setdiff(1:ℓ(s),c)
    α  = d


    s1         = [  sum(abs.( H(Σi;j)^(1/2) \ εi ))      for (εi,Σi) in zip(εs,Σs) ] 
    s1         = s1 / maximum(s1[c]) # rescale ( equivalent to letting α be a percent of the range)
    B          = quantile(s1,q)
    s1[s1.>B] .= B

    # quantile
    λ = n/(n+1) * mean(s1[c]) / maximum( [j, (α - B/(n+1))])

    println(λ)
    println()

    # coverage
    cwcov_l = err_componentwise_coverage( εs[t], Σs[t], λ  )
    cwcov_u = err_componentwise_coverage_upper( εs[t], Σs[t], λ  )
    mvcov   = mean(  s[t] .< λ )
    sanity  = mean( s1[t]/λ ) 


    # volume
    v = [ exp(  (1/Base.size(ss,1))*log_V_ellipsoid(Base.size(ss,1), λ, H(ss ;j))  )   for ss in Σs[t] ]
    
    return [ d, q, λ, cwcov_l, cwcov_u, mvcov, mean(v), median(v), maximum(v), sanity  ]
end

function riskcon_robust_quantile( s_orig, Σs, εs ; d, q, cal_frac=0.6, j=1e-8 )

    # threshold score
    lg = size.(Σs,1)
    s  = copy(s_orig) ./ lg
    n  = r(ℓ(s)*cal_frac)
    c  = 1:n
    t  = setdiff(1:ℓ(s),c)
    α  = d

    sq         = [  quantile( vec( abs.(H(Σi;j)^(1/2) \ εi) ) ,  0.9 )      for (εi,Σi) in zip(εs,Σs) ]
    sq         = sq ./ sqrt.( lg )
    Σs         = copy(Σs) .* lg
    sq         = sq / maximum(sq[c])  # recale ( equivalent to letting α be a percent of the range)
    B          = quantile(sq,q)
    sq[sq.>B] .= B

    # quantile
    λ = n/(n+1) * mean(sq[c]) / maximum([j, (α - B/(n+1)) ])

    # coverage
    cwcov_l = err_componentwise_coverage( εs[t], Σs[t], λ  )
    cwcov_u = err_componentwise_coverage_upper( εs[t], Σs[t], λ  )
    mvcov   = mean(  s[t] .< λ )
    sanity  = mean( sq[t]/λ  )

    println( α - B/(n+1) )
    println(λ)
    println()

    # volume
    v = [ exp(  (1/size(ss,1))*log_V_ellipsoid(size(ss,1), λ, H(ss ;j))  )   for ss in Σs[t] ]

    return [ d, q, λ, cwcov_l, cwcov_u, mvcov, mean(v), median(v), maximum(v), sanity ]
end



# ---------------------------------------------------------------
# ===============================================================
#
# Set up test
#
# ===============================================================
# ---------------------------------------------------------------

nT           = parse( Int64, ARGS[1] )
seed         = 12
files        = [ path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
task         =  "dia"
inds         = 329:713
readpath     = path*"/data/covariance/force_" 
path         = path*"/data/force_energy/"

# check that the necessary precomputed files exist
C    = CSV.read( readpath*"C.csv", DataFrame)[:,1]
s    =        CSV.read( readpath*"sor.csv", DataFrame)[:,1]
name = string( readpath, "SOR_", length(s), "_", length(C) )
!isfile( name*".csv"   )  &&    error( "No precomputed force SOR covariance file was located."  )
!isfile( name*"_w.csv" )  &&    error( "No precomputed force SOR weights file was located."  )


# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [inds], stresses  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )
predict( xyz, settings ; task, path, readpath, seed, nT )

