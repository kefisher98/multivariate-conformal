using LinearAlgebra
using Random
using CSV
using DataFrames
using Distributions
using SpecialFunctions


# import local files
path  = @__DIR__
path  = path[1:end-4]
include( path*"/src/modules/auxiliary.jl")
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")

# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------


# record: make folder with time stamp, make multiple .csv files

# parameters to vary: ϱ, prop_scale, p, ϕ
function run_tests(  ; n=100, n0=1000, np=1000, d=100, p=1, s=100, v=1, ϱ=0.9, ϱ0=0.9,  ϕ=identity, σ²=1e-4, seed=12, θ=1e-8, ζ=4, j=1e-8,
                       α_set=collect(0.4:0.05:0.95), prop_n=10, prop_scale=10,  prop_dims=collect(5:5:100), path   )

    file = init( n, s, ϱ/v, prop_scale, path )

    # generate covariates
    Random.seed!(seed+1)
    X  = generate_features( n  ; d, s=1, v, ϱ    ) 
    X₀ = generate_features( n0 ; d, s,   v, ϱ=ϱ0 )
    Xₚ = generate_features( np ; d, s,   v, ϱ    )

    # data generation
    Y  = [ observe( x̃ ; d=[d,p,1], ϕ, σ²,   seed=seed+2 )  for x̃ in X  ]
    Y₀ = [ observe( x̃ ; d=[d,p,1], ϕ, σ²=0, seed=seed+2 )  for x̃ in X₀ ]
    Yₚ = [ observe( x̃ ; d=[d,p,1], ϕ, σ²=0, seed=seed+2 )  for x̃ in Xₚ ]

    # predict
    ε, μ, Σ  = train( hcat(X...), vcat(Y...), [X₀;Xₚ], [Y₀;Yₚ] ; θ, ζ, j )

    # calibrate and push forward
    univariate(   μ[1:n0], Σ2σ.(Σ[1:n0]), Y₀, μ[n0+1:end], Σ2σ.(Σ[n0+1:end]), Yₚ, ε ; file, path, prop_n, prop_scale, prop_dims, seed, α_set )
    multivariate( μ[1:n0],      Σ[1:n0],  Y₀, μ[n0+1:end],      Σ[n0+1:end],  Yₚ, ε ; file, path, prop_n, prop_scale, prop_dims, seed, α_set, j )
    prop_multi(   μ[1:n0],      Σ[1:n0],  Y₀, μ[n0+1:end],      Σ[n0+1:end],  Yₚ, ε ; file, path, prop_n, prop_scale, prop_dims, seed, α_set, j )
end

function init( n, s, c, prop_scale, path )
    file = string( "n", n, "_s", s, "_cor", c, "_scale", prop_scale, ".csv" )
    hcw  = [:seed, :alpha, :error, :prop_dim, :cw_coverage_0, :mv_coverage_0, :volume_0, :cw_coverage_1, :mv_coverage_1, :volume_1 ]
    hpc  = [:seed, :alpha, :error, :prop_dim, :cw_coverage_lo_1, :cw_coverage_hi_1, :mv_coverage_1, :volume_1 ]
    hmv  = [:seed, :alpha, :error, :prop_dim, :cw_coverage_lo_0, :cw_coverage_hi_0, :mv_coverage_0, :volume_0, :cw_coverage_lo_1, :cw_coverage_hi_1, :mv_coverage_1, :volume_1 ]

    makeFile( hcw, string(path, "cw_",   file) )
    makeFile( hmv, string(path, "mv_",   file) )
    makeFile( hpc, string(path, "pc_",   file) )

    return file
end


# ---------------------------------------------------------------
# ===============================================================
#
# Data generation
#
# ===============================================================
# ---------------------------------------------------------------

function generate_features( n ; d, s, v, ϱ )
    m = zeros(s)
    C = (v-ϱ)*I + ϱ*ones(s,s)
    return  [ permutedims( rand( MvNormal(m,C), d ), [2,1] )  for i in 1:n ]
end



function observe( X ; d=[100,200,1], ϕ=erf, σ²=0., seed=1  )
    Random.seed!(seed)
    M = X
    for i in 1:length(d)-2
        Θ = rand( Normal(), d[i], d[i+1] ) / sqrt(d[i])
        M = ϕ.(Θ'*M)
    end
    W = rand( Normal(), d[end-1], d[end] ) / sqrt(d[end-1])

    # generate noise, different randomness from features
    Random.seed!(MersenneTwister(1234))
    η = sqrt(σ²) * rand( Normal(), size(X,2) ) 

    return M'*W + η
end



# ---------------------------------------------------------------
# ===============================================================
#
# Training
#
# ===============================================================
# ---------------------------------------------------------------


function train( X, Y, X₀, Y₀ ; θ, ζ=1, j=1e-8 ) 

    K = (X'*X).^ζ
    w = ( H(K;j) + θ*I ) \ Y

    μ = []
    Σ = []
    for x̃ in X₀
        Ks  = (X'*x̃).^ζ
        Kss = (x̃'*x̃).^ζ
        push!( μ, Ks'*w )
        push!( Σ, Kss - Ks'*( ( H(K;j) + θ*I ) \ Ks )  )
    end

    return rmse( squish(μ), squish(Y₀) ), μ, Σ
end



# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------


function univariate(  μc, σc, Yc, μp, σp, Yp, ε ; file, path, prop_n, prop_scale, prop_dims, α_set, seed, cal_frac=0.5 )

    # calibration scores
    sc   = [ abs.( m - y ) ./ σ   for (m,σ,y) in zip(μc, σc, Yc) ]
    n    = length( squish(sc) ) 
    nSet = length(sc[1])
    
    for α in α_set 
        q = quantile( squish(sc), minimum( [1., ceil( (n+1)*α )/n]) )

        # original space coverage and volume
        cwcov = componentwise_coverage( squish(μp), q*squish(σp), squish(Yp) )
        mvcov = set_coverage( squish(μp), q*squish(σp), squish(Yp)  ; nSet )
        vol0  = mean( exp.( (1/ℓ(μc[1]))*sum(log.( 2q*hcat(σp...) ) ; dims=1 )) )

        for d in prop_dims
            # propagated space
            mvc = 0
            cwc = 0
            vol = 0
            for i in 1:prop_n
                M    = rand( Normal(), d, nSet ) / prop_scale
                Me   = [ abs.( M*(m-y) )   for (m,y) in zip(μp,Yp) ]
                Mσ   = [     q*M*S         for S     in σp         ]  

                # [μ-σ, μ+σ]
                # maximum.(M*S) - minimum.(M*S)
                mvc += mean( [ prod(e .< maximum(abs.(ms)))           for (ms,e) in zip(Mσ,Me) ]) / prop_n 
                cwc += mean( [ mean(e .< maximum(abs.(ms)))           for (ms,e) in zip(Mσ,Me) ]) / prop_n
                vol += mean( [ exp( (1/d)*sum(log.(  2*abs.(ms) )))   for  ms    in     Mσ     ]) / prop_n
            end

            addToFile( [ seed  α  ε  d  cwcov  mvcov  vol0  cwc  mvc  vol ], string(path,"cw_",file) )
        end
    end
end


# ---------------------------------------------------------------
# ===============================================================
#
# Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------

function multivariate( μc, Σc, Yc, μp, Σp, Yp, ε ; file, path, prop_n, prop_scale, prop_dims, α_set, seed, j )
    

    # calibration scores
    sc = [ dot( yi-μi,  H(Σi;j) \(yi-μi) )       for (yi,μi,Σi) in zip(Yc,μc,Σc) ]

    for α in α_set
        q = quantile( sc, minimum([1., ceil( (ℓ(sc)+1)*α )/ℓ(sc)])  )
        
        # original space coverage and volume
        cwcov_lo = componentwise_coverage( μp, Σp, Yp, q )
        cwcov_up = componentwise_coverage_upper( μp, Σp, Yp, q )
        mvcov    = set_coverage( μp, Σp, Yp, q )
        vol0     = mean( [ exp((1/size(S,1))*log_V_ellipsoid(size(S,1), q, H(S;j)))  for S in Σp ])

        for d in prop_dims
            # propagated space
            mvc = 0
            cwu = 0
            cwl = 0
            vol = 0
            for i in 1:prop_n
                M    = rand( Normal(), d, length(Yp[1]) ) / prop_scale

                sp   = [ dot( M*(yi-μi),  (M*H(Σi;j)*M') \ (M*(yi-μi) ))             for (yi,μi,Σi) in zip(Yp,μp,Σp) ]
                mvc += mean( sp .<= q ) / prop_n
                 
                sp   = vcat([ ( M*(yi-μi) ).^2 /  diag(M*H(Σi;j)*M')            for (yi,μi,Σi) in zip(Yp,μp,Σp) ]...)
                cwu += mean( sp .<= q ) / prop_n

                sp   = vcat([ ( M*(yi-μi) ).^2 .*  diag( inv(M*H(Σi;j)*M') )         for (yi,μi,Σi) in zip(Yp,μp,Σp) ]...)
                cwl += mean( sp .<= q ) / prop_n
                
                vol += mean( [ exp( (1/d)*log_V_ellipsoid(size(S,1), q, M*H(S;j)*M'  ))  for  S in Σp ] ) /  prop_n
            end

            addToFile( [ seed  α  ε  d  cwcov_lo cwcov_up  mvcov  vol0  cwl cwu  mvc  vol ], string(path,"mv_",file) )
        end
    end
end


# ---------------------------------------------------------------
# ===============================================================
#
# Propagation-corrected Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------

function prop_multi( μc, Σc, Yc, μp, Σp, Yp, ε ; file, path, prop_n, prop_scale, prop_dims, α_set, seed, j )

    for d in prop_dims
        # propagated space
        mvc = zeros(length(α_set))
        cwu = zeros(length(α_set))
        cwl = zeros(length(α_set))
        vol = zeros(length(α_set))

        for i in 1:prop_n
            M    = rand( Normal(), d, length(Yp[1]) ) / prop_scale
            sc = [ dot( M*(yi-μi),  (M*H(Σi;j)*M')\ (M*(yi-μi)) )       for (yi,μi,Σi) in zip(Yc,μc,Σc) ]

            spmv  = [ dot( M*(yi-μi),  (M*H(Σi;j)*M') \ (M*(yi-μi) ))        for (yi,μi,Σi) in zip(Yp,μp,Σp) ]
            spcwu = vcat([ ( M*(yi-μi) ).^2 /  diag(M*H(Σi;j)*M')            for (yi,μi,Σi) in zip(Yp,μp,Σp) ]...)
            spcwl = vcat([ ( M*(yi-μi) ).^2 .*  diag( inv(M*H(Σi;j)*M') )         for (yi,μi,Σi) in zip(Yp,μp,Σp) ]...)
            for (j,α) in enumerate(α_set)
                q    = quantile( sc, minimum([1., ceil( (ℓ(sc)+1)*α )/ℓ(sc)])  )

                mvc[j] += mean( spmv  .<= q ) / prop_n
                cwu[j] += mean( spcwu .<= q ) / prop_n
                cwl[j] += mean( spcwl .<= q ) / prop_n
                vol[j] += mean( [ exp( (1/d)*log_V_ellipsoid(size(S,1), q, M*H(S;j)*M'  ))  for  S in Σp ] ) /  prop_n
            end

        end
        for (j,α) in enumerate(α_set)
            addToFile( [ seed  α  ε  d  cwl[j] cwu[j]  mvc[j]  vol[j] ], string(path,"pc_",file) )
        end
    end
end


# ---------------------------------------------------------------
# ===============================================================
#
# Set up test
#
# ===============================================================
# ---------------------------------------------------------------

ϱ          = parse(Float64, ARGS[1] )
ϱ0         = parse(Float64, ARGS[2] )
path       = path*"/data/synthetic/"
n          = 100
np         = 1000
ϕ          = identity
seed       = 12
θ          = 1e-8
prop_n     = 10
prop_scale = 10

run_tests( ; path, ϱ, ϱ0, n, np, ϕ, seed, θ, prop_n, prop_scale )

