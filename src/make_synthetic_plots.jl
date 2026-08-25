using Plots
using LinearAlgebra
using Random
using Distributions
using CSV
using DataFrames
using LaTeXStrings
using Plots.Measures

path = @__DIR__
path = path[1:end-3]
include(path*"src/modules/auxiliary.jl")

# Set Plot parameters
titlefont   = font("Computer Modern",16);
guidefont   = font("Computer Modern",16);
legendfont  = font("Computer Modern",14);

size      = (450,450)  
lw        = 2
prefix    = ""
suffix    = "_equalaxis"
extension = ".pdf"

# ---------------------------------------------------------------------------------------
# =======================================================================================
# Synthetic
# =======================================================================================
# ---------------------------------------------------------------------------------------

# propagated coverage by propagation dim
function alpha_prop_cov( data, prop_dims ; size=size, sc=:roma, bump=1, select=:, title="", freq=5, legend )
    
    fig = plot( ; legendfont, guidefont, titlefont, size, xlabel=L"$1-\alpha$", ylabel="multivariate coverage", title, legend  )
    c   = cgrad( sc, length(prop_dims)+bump, categorical = true )[select]


    plot!([0,1],[0,1],seriestype=:straightline,label="x=y",lw=3,color=:gray,linestyle=:dot)

    for (i,p) in enumerate(prop_dims)
        label = mod(i,freq)==0 ? label=string("prop. dim. = ",p) : false

        a = multifilter( data, [:prop_dim], [(p,)] )
        plot!( a[:,:alpha], a[:,:mv_coverage_1] ; color=c[i], lw=3, label )
    end
    plot!(xlims=(0.4,0.95),ylims=(0.0,1.01))
    return fig
end

function scatter_coverage_volume( mv, cw, prop_dims ; arrows=false, size=size, title="", lbl=[1,20], sca=:oslo, scb=:Reds_9, bump=1, select=:, legend, freq=5, col=:volume_1, alpha=0.7 )


    fig = plot( ; legendfont, guidefont, titlefont, size, xlabel="multivariate coverage", ylabel=L"(set volume)$^{1/d}$", title, legend  )
    #ca  = reverse(cgrad( sca, length(prop_dims)+bump, categorical = true )[select])
    cb  = reverse(cgrad( scb, length(prop_dims)+bump, categorical = true )[select])

    ca  = cgrad( sca, length(prop_dims)+bump, categorical = true )[select]
    #cb  = cgrad( scb, length(prop_dims)+bump, categorical = true )[select]

    #for (i,p) in enumerate(prop_dims)
    for (i,p) in enumerate(reverse(prop_dims))
        b = multifilter( cw, [:prop_dim], [(p,)] )
        scatter!( b[:,:mv_coverage_1], b[:,col] ; color=cb[i], markerstrokecolor=:crimson, markershape=:square, alpha, markersize=7, label=false )
    end

    #for (i,p) in enumerate(prop_dims)
    for (i,p) in enumerate(reverse(prop_dims))
        a = multifilter( mv, [:prop_dim], [(p,)] )
        scatter!( a[:,:mv_coverage_1], a[:,col] ; color=ca[i], markerstrokecolor=:slateblue, alpha, markersize=7, label=false )
    end
    #########################
    # arrow
    #########################

    if arrows
        x_arrow = [0.42, 0.62, 0.62, 0.7, 0.62, 0.62, 0.42]
        y_arrow = [0.4, 0.4, 0.38, 0.42, 0.46, 0.44, 0.44] 
        plot!(
            Shape(x_arrow, y_arrow),
            fillcolor = :orange,
            fillalpha = 0.8,       # Transparency of the interior fill (0.0 = invisible, 1.0 = solid)
            linecolor = :orange,
            linealpha = 0.8,       # Transparency of the border outline
            linewidth = 1.5,
            label = false
        )
        x_arrow = [0.36, 0.4,  0.68, 0.7,  0.715, 0.62,  0.645] .- 0.05
        y_arrow = [0.345, 0.38,  0.1, 0.12, 0.04, 0.04, 0.065] .- 0.05
        plot!(
            Shape(x_arrow, y_arrow),
            fillcolor = :green,
            fillalpha = 0.2,       # Partially transparent
            linecolor = :darkgreen,
            linewidth = 2,
            linealpha = 0.9,
            label = false
        )
    end

    #########################
    # labels
    #########################

    #ca  = cgrad( sca, length(prop_dims)+bump, categorical = true )[select]
    cb  = cgrad( scb, length(prop_dims)+bump, categorical = true )[select]
    ca  = reverse(cgrad( sca, length(prop_dims)+bump, categorical = true )[select])
    #cb  = reverse(cgrad( scb, length(prop_dims)+bump, categorical = true )[select])

    for (i,p) in enumerate(prop_dims)
    #for (i,p) in enumerate(reverse(prop_dims))
        b = multifilter( cw, [:prop_dim], [(p,)] )
        label_b = i in lbl ? label=string("uni,    dim. = ",p) : false
        scatter!( [], []  ; color=cb[i], markerstrokecolor=:crimson, markershape=:square, alpha, markersize=7, label=label_b )
    end

    for (i,p) in enumerate(prop_dims)
    #for (i,p) in enumerate(reverse(prop_dims))
        a = multifilter( mv, [:prop_dim], [(p,)] )
        label_a = i in lbl ? label=string("multi, dim. = ",p) : false
        scatter!( [], [] ; color=ca[i], markerstrokecolor=:slateblue, alpha, markersize=7, label=label_a )
    end

    return fig
end

# read data
cw_03 = CSV.read(path*"/data/synthetic/cw_n100_s100_cor0.3_scale10.csv",DataFrame)
mv_03 = CSV.read(path*"/data/synthetic/mv_n100_s100_cor0.3_scale10.csv",DataFrame)
cw_09 = CSV.read(path*"/data/synthetic/cw_n100_s100_cor0.9_scale10.csv",DataFrame)
mv_09 = CSV.read(path*"/data/synthetic/mv_n100_s100_cor0.9_scale10.csv",DataFrame)
pc_09 = CSV.read(path*"/data/synthetic/pc_n100_s100_cor0.9_scale10.csv",DataFrame)
mv_99 = CSV.read(path*"/data/synthetic/mv_n100_s100_cor0.99_scale10.csv",DataFrame)
pc_99 = CSV.read(path*"/data/synthetic/pc_n100_s100_cor0.99_scale10.csv",DataFrame)

prop_dim = unique(cw_03[:,:prop_dim])

# coverage vs volume
fig_03 = scatter_coverage_volume( mv_03, cw_03, prop_dim ; title=L"(a)   $\rho=0.3$", legend=:topleft, arrows=true )
fig_09 = scatter_coverage_volume( mv_09, cw_09, prop_dim ; title=L"(b)   $\rho=0.9$", legend=:topleft, arrows=false )
savefig( fig_03, string( path, "figures/synthetic/", prefix, "synthetic_cov_vol_0.3", suffix, extension )  )
savefig( fig_09, string( path, "figures/synthetic/", prefix, "synthetic_cov_vol_0.9", suffix, extension )  )

# propagation coverage
fig_cw_03 = alpha_prop_cov( cw_03, prop_dim ; title=L"(a)  uni-conformal: $\rho=0.3$",   legend=false )
fig_mv_03 = alpha_prop_cov( mv_03, prop_dim ; title=L"(b)  multi-conformal: $\rho=0.3$", legend=:bottomright )
fig_cw_09 = alpha_prop_cov( cw_09, prop_dim ; title=L"(c)  uni-conformal: $\rho=0.9$",   legend=false )
fig_mv_09 = alpha_prop_cov( mv_09, prop_dim ; title=L"(d)  multi-conformal: $\rho=0.9$", legend=:bottomright )
fig_pc_09 = alpha_prop_cov( pc_09, prop_dim ; title=L"prop-multi-conformal: $\rho=0.9$", legend=:bottomright )
savefig( fig_cw_03, string( path, "figures/synthetic/", prefix, "synthetic_cw_prop_cov_0.3", suffix, extension )  )
savefig( fig_mv_03, string( path, "figures/synthetic/", prefix, "synthetic_mv_prop_cov_0.3", suffix, extension )  )
savefig( fig_cw_09, string( path, "figures/synthetic/", prefix, "synthetic_cw_prop_cov_0.9", suffix, extension )  )
savefig( fig_mv_09, string( path, "figures/synthetic/", prefix, "synthetic_mv_prop_cov_0.9", suffix, extension )  )
savefig( fig_pc_09, string( path, "figures/synthetic/", prefix, "synthetic_pc_prop_cov_0.9", suffix, extension )  )

# correlation misspecification
fig_mv = alpha_prop_cov( mv_09, prop_dim ; title="(a)   multi-conformal", legend=:bottomright )
fig_pc = alpha_prop_cov( pc_09, prop_dim ; title="(b)   prop-multi-conformal", legend=:bottomright )
savefig( fig_mv, string( path, "figures/synthetic/", prefix, "synthetic_mv_prop_cov_0.99_mispec", suffix, extension )  )
savefig( fig_pc, string( path, "figures/synthetic/", prefix, "synthetic_pc_prop_cov_0.99_mispec", suffix, extension )  )
