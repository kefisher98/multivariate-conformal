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
using Plots, Plots.Measures
using LaTeXStrings, Plots

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

# ---------------------------------------------------------------
# ===============================================================
#
# Outer loop
#
# ===============================================================
# ---------------------------------------------------------------

function extract(m, C, y; nConfig=24, nT, nCal)
    iCal = [ sample(1:nT, nConfig) for _ in 1:nCal ]
    return [m[i] for i in iCal], [C[i,i] for i in iCal  ], [y[i] for i in iCal]
end

# record: make folder with time stamp, make multiple .csv files

function elastic_constant_from_energy( α, config, xyz, seed_set ;  n_tune=20, fold=20, tune_seed=1, tune_set,
                                       settings, j=1e-8, ε=0.05, h=0.05, task="dia", σ²=1e-8, v=1.0, core=350, target=100 )


    # draw configuration indices
    ec  = []
    cw  = []
    cb  = []
    mv  = []
    for seed in seed_set
        # choose training and test set
        Random.seed!(tune_seed);
        S  = split( sample( tune_set, n_tune, replace=false  ), r(n_tune/fold) )

        # draw configuration indices
        Random.seed!(seed);
        T      =      sample( setdiff( xyz[task].inds, [config;S]   ), target, replace=false )       # test
        C      = [1 ; sample( setdiff( xyz[task].inds, [config;T;S] ), core,   replace=false )   ]   # train
        nC, nT = length(C), length(T)

        # train GP
        w, μ, Σ, xC, K, eC, eT, err, ρ = train( task, xyz ; settings, C, T, σ², v )
        
        # conformal scores
        qcw      = component_wise( μ, Σ2σ(Σ), eT ; α ) 
        qcb      = component_wise( μ, Σ2σ(Σ), eT ; α, bonf=true ) 
        μ, Σ, eT = extract(  μ, Σ, eT ; nT, nCal=nT*2 )
        qmv      = multivar( μ, Σ, eT ; α, j )       
        
        # predict elastic constant
        sd   = settings.descriptor
        desc = SOAPDescriptor(  species=sd.species, n_max=sd.n_max, l_max=sd.l_max , sigma=sd.sigma, r_cut=sd.r_cut )
        
        # predict energies at perturbed geometries
        errC, rr, KC = perturb_relax( config, w, xC, K ; desc, xyz, task, v, σ², h, ε, nC, position=2 )
        #errC, rr, KC = perturb_relax( config, w, xC, K ; desc, xyz, task, v, σ², h, ε, nC, position=1 )

        push!( ec,  errC )
        push!( cw,  qcw*rr )
        push!( cb,  qcb*rr )
        push!( mv,  qmv*KC )
    end
    return ec, cw, cb, mv
end


# ---------------------------------------------------------------
# ===============================================================
#
# Training
#
# ===============================================================
# ---------------------------------------------------------------

function train( task, xyz ; settings, C, T, σ²=0.001, v=10000.0  )

    # construct covariance structures
    xC, K, Ks, Kss, eC, eT = covariance( task, xyz ; settings, C, T)

    # train GP 
    w = (v*H(K) + σ²*I) \ eC
    μ = (v*Ks)'*w
    Σ = v*Kss - (v*Ks)'*( (v*H(K) + σ²*I) \  (v*Ks) )

    # report error
    err = rmse(μ,eT)
    ρ   =  cor(μ,eT)

    return w, μ, (Σ+Σ')/2, xC, K, eC, eT, err, ρ
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
                          Optim.Options( g_tol=1e-1, x_abstol=1e-6, iterations=100, show_trace=false, store_trace=true, extended_trace=true ))

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

function perturb_relax( config, w, xC, K ; desc, xyz, task, v, σ², h, ε, nC, position,  nS=8, j=1e-10  )

    # apply perturbation
    strain   = h*[1 0 0; 0 0 0.5; 0 0.5 0]
    local_r  = Xoshiro(1111)
    kick     = 0.001* rand(local_r,Normal(),3,3)
    kick     = (kick+kick')/2
    config_a = ase.read( xyz[task].file, index=string(config-1) )
    
    G, s, cell, pbc = atom_extract(config_a)
    config_b        = atom_create( G*(I+kick), s, cell*(I + kick), pbc )

    if position in [1,2,3]
        xS_a, Vm_a = perturb_relax( config_a, ε*diagm(ej(position)), strain ; desc, xC, w, v )
        xS_b, Vm_b = perturb_relax( config_b, ε*diagm(ej(position)), strain ; desc, xC, w, v )
    else
        if position in [4,5]
            j = 1
            i = position - 2 
        else
            j = 2
            i = 3
        end
        xS_a, Vm_a = perturb_relax( config_a, 0.5*ε*(ej(i)*ej(j)' + ej(j)*ej(i)'), strain ; desc, xC, w, v )
        xS_b, Vm_b = perturb_relax( config_b, 0.5*ε*(ej(i)*ej(j)' + ej(j)*ej(i)'), strain ; desc, xC, w, v )
    end

    
    # predict energies and elastic constants
    Ks  = covariance(  [xS_a; xS_b; xC], 1:nS, nS.+(1:nC) ; C=energy )
    Kss = covariance(  [xS_a; xS_b],     1:nS,      1:nS  ; C=energy )

    # infer energies and covariances
    μE = v*Ks * w
    KE = v*Kss - (v*Ks)*( (v*K + σ²*I) \  (v*Ks') )

    # propagation operator
    M = zeros(2,8)
    M[1,1:4] = [1 -1 -1  1]  ./ (4ε*h*Vm_a')
    M[2,5:8] = [1 -1 -1  1]  ./ (4ε*h*Vm_b')

    # propagate to elastic constant prediction
    y  = xyz[task].EC[position]
    μC = M*μE
    KC = M*KE*M'
    rr =  reshape(sum( reshape( Σ2σ(KE),4,2) ./ (4ε*h*[Vm_a Vm_b])   ; dims=1 ),:,1)

    return y.-μC, rr, KC
end




# ---------------------------------------------------------------
# ===============================================================
#
# Component-wise conformal
#
# ===============================================================
# ---------------------------------------------------------------


function component_wise( μ, σ, y ; α, cal_frac=0.5, nSet=8, bonf=false  )

    # calibration scores
    αb = bonf ? 1 - (1-α)/nSet : α
    s = abs.( μ - y ) ./ σ

    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum( [1, ceil( (ℓ(c)+1)*αb )/ℓ(c) ]) )

    return q
end



# ---------------------------------------------------------------
# ===============================================================
#
# Multivariate conformal
#
# ===============================================================
# ---------------------------------------------------------------


function multivar( μ, Σ, y ; α, cal_frac=0.5, j )

    # compute score
    s = [ (yi-μi)'*( H(Σi;j) \(yi-μi))   for (yi,μi,Σi) in zip(y,μ,Σ) ]
    
    # quantile
    c = 1:r(ℓ(s)*cal_frac)
    t = setdiff(1:ℓ(s),c)
    q = quantile( s[c], minimum( [1, ceil( (ℓ(c)+1)*α )/ℓ(c) ] ) )

    return q
end



# ---------------------------------------------------------------
# ===============================================================
#
# Plot sets
#
# ===============================================================
# ---------------------------------------------------------------

function plot_ellipsoids2d(Cs::Vector{<:AbstractMatrix}, x0s::Vector{<:AbstractVector}, rs::Union{Real, Vector{<:Real}}; num_points::Int=100)
    n = length(Cs)
    if length(x0s) != n
        error("The number of matrices in Cs ($n) must match the number of centers in x0s ($(length(x0s))).")
    end

    radii = rs isa Real ? fill(rs, n) : rs
    if length(radii) != n
        error("The length of rs must match the number of ellipsoids ($n).")
    end

    p = plot(aspect_ratio=:equal, grid=true)

    θ = range(0, 2π, length=num_points)
    circle_points = [cos.(θ)'; sin.(θ)'] # 2 x num_points matrix

    for i in 1:n
        C = Cs[i]
        x0 = x0s[i]
        r = radii[i]

        if size(C) != (2, 2) || !issymmetric(C)
            error("Matrix C at index $i must be a symmetric 2x2 matrix.")
        end

        # Eigendecomposition
        eg = eigen(C)
        λ = eg.values
        V = eg.vectors

        if any(λ .<= 0)
            error("Matrix C at index $i must be positive-definite.")
        end

        # Calculate semi-axes
        a = sqrt(r * λ[1])
        b = sqrt(r * λ[2])

        # Transform the circle to an ellipsoid
        scaling_matrix = [a 0; 0 b]
        aligned_ellipse = scaling_matrix * circle_points
        ellipsoid_points = V * aligned_ellipse .+ x0

        X = ellipsoid_points[1, :]
        Y = ellipsoid_points[2, :]

        plot!(p, X, Y, label=false, lw=0.5 ; color=:slateblue)
    end

    scatter!( [0.], [0.], color=:black, markerstrokecolor=:black, markersize=8, label=false )

    return p
end

function plot_rectangles2d(ys::Vector{<:AbstractVector}, y0s::Vector{<:AbstractVector}, color)
    
    n = length(ys)
    if length(y0s) != n
        error("The number of dimension vectors ($n) must match the number of centers ($(length(y0s))).")
    end

    p = plot(aspect_ratio=:equal, grid=true)

    for i in 1:n
        y = ys[i]
        y0 = y0s[i]

        if length(y) != 2 || length(y0) != 2
            error("Vectors at index $i must have exactly 2 components.")
        end

        w, h = y[1], y[2]
        cx, cy = y0[1], y0[2]

        if w <= 0 || h <= 0
            error("Rectangle dimensions must be positive. Error at index $i.")
        end

        hw = w / 2
        hh = h / 2

        X_local = [hw, -hw, -hw,  hw, hw]
        Y_local = [hh,  hh, -hh, -hh, hh]

        X = X_local .+ cx
        Y = Y_local .+ cy

        plot!(p, X, Y, label=false, lw=0.5 ; color)
    end

    scatter!( [0.], [0.], color=:black, markerstrokecolor=:black, markersize=8, label=false )
    return p
end



# ---------------------------------------------------------------
# ===============================================================
#
# Set up test
#
# ===============================================================
# ---------------------------------------------------------------

α            = 0.75
tune_seed    = 1
tune_set     = setdiff( 225:713, [225,329,549,659] )
seed_set     = 2:21
files        = [ path*"/data/configurations/GAP_silicon.xyz"]
key          =   path*"/data/configurations/key_structure.csv"
quantities   = ["dft_energy"]
derivatives  = ["dft_force"]
stresses     = ["dft_virial"]
task         =  "dia"
inds         =  225:713
config       = 329

#   2 atoms: 225:328
#  16 atoms: 329:548
#  54 atoms: 549:658
# 128 atoms: 659:713


# run experiment
xyz          = set_xyz( files, quantities, derivatives, key, [inds], stresses  )
descriptor   = set_SOAP( ; species=["Si"], r_cut=5 )
settings     = set_settings( ;  descriptor )


err, cw, cb, mv = elastic_constant_from_energy( α, config, xyz, seed_set ; settings, task, tune_set, tune_seed )


titlefont   = font("Computer Modern",16)
guidefont   = font("Computer Modern",16);
legendfont  = font("Computer Modern",14)


# project to 2D
inds   = [1,2]
ss     = 2:2:20
shift  = [0.25,0.5,0.75]
ylims  = (-2.7,2.7)
xlims  = (-2.7,2.7)
xlabel = L"original $C_{12}$ [eV/$\AA^3$]" 
ylabel = L"perturbed $C_{12}$ [eV/$\AA^3$]"
ess    = [  x[inds]                            for x in err[ss] ]
mvs    = [ (x[inds,inds] + x[inds,inds]')/2    for x in mv[ss]  ]
cbs    = [  x[inds]                            for x in cb[ss]  ]
cws    = [  x[inds]                            for x in cw[ss]  ]

# multivariate plot
mv_fig = plot_ellipsoids2d( mvs, ess, ones(length(mvs)) )
mv_fig = plot!( mv_fig, title="multivariate conformal"; guidefont, titlefont, legendfont )
mv_fig = plot!( mv_fig ; xticks=[-1,1], yticks=[-1,1], xlims, ylims, grid=false, xlabel, ylabel, margin=2mm )
mv_fig = scatter!( mv_fig, xlims[1].+shift, ylims[2].-shift[1]*ones(3) ; label=false, markersize=10, color=:slateblue, markerstrokecolor=:slateblue, marker=:star )


# bonferroni plot
bf_fig = plot_rectangles2d( cbs, ess, :sienna )
bf_fig = plot!( bf_fig, title="univariate Bonferroni" ; guidefont, titlefont, legendfont )
bf_fig = plot!( bf_fig ; xticks=[-1,1], yticks=[-1,1], xlims, ylims, grid=false, xlabel, ylabel, margin=2mm ) 
bf_fig = scatter!( bf_fig, [-10000,-100001], [-10000,-100001] ; label="truth", markersize=6, color=:black, markerstrokecolor=:black )
bf_fig = plot!(bf_fig  ; legend=:bottomright )
bf_fig = plot!( bf_fig, [-10000,-100001], [-10000,-100001] ; label="prediction set", lw=2, color=:black )
bf_fig = scatter!( bf_fig, xlims[1].+shift[1:2], ylims[2].-shift[1]*ones(2) ; label=false, markersize=10, color=:sienna, markerstrokecolor=:sienna, marker=:star )

# univariate plot
cw_fig = plot_rectangles2d( cws, ess, :goldenrod3)
cw_fig = plot!( cw_fig, title="univariate conformal" ; guidefont, titlefont, legendfont )
cw_fig = plot!( cw_fig  ; xticks=[-1,1], yticks=[-1,1], xlims, ylims, grid=false, xlabel, ylabel, margin=2mm  )
cw_fig = scatter!( cw_fig, xlims[1].+shift[1:1], ylims[2].-shift[1]*ones(1) ; label=false, markersize=10, color=:goldenrod3, markerstrokecolor=:goldenrod3, marker=:star )


three = plot( cw_fig, bf_fig, mv_fig, layout=(1,3), size=(1200,400),margin=8mm )
savefig( three, path*"/figures/energy_elastic_constant/intro_fig.pdf" )
