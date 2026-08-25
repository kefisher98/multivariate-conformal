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
# Forces and energies
# =======================================================================================
# ---------------------------------------------------------------------------------------

function multiconformal_guarantees( α_set, mv, cw ; size=size, titles )
    mv_fig = plot( ; legendfont, guidefont, titlefont, size,  ylabel="multivariate coverage", xlabel=L"$1-\alpha$", title=titles[1]  )
    plot!( α_set, mv[:,:mv] ; color=:slateblue, alpha=0.9, lw=3, markersize=8, markershape=:circle, markerstrokecolor=:slateblue, label="multivariate")
    plot!( α_set, cw[end-11:end,:mv] ; color=:goldenrod3, alpha=0.9,  lw=3, markersize=8, markershape=:dtriangle, markerstrokecolor=:goldenrod3, label="univariate")
    plot!([0,1],[0,1],seriestype=:straightline,label=false,lw=3,color=:gray,linestyle=:dot)
    plot!(ylims=(-0.03,1.03))

    cw_fig = plot( ; legendfont, guidefont, titlefont, size, ylabel="univariate coverage", xlabel=L"$1-\alpha$", title=titles[2]  )
    plot!( α_set, mv[:,:cw_hi] ; color=:slateblue, alpha=0.9,  lw=3, markersize=8, markershape=:circle, markerstrokecolor=:slateblue, label="multivariate")
    plot!( α_set, cw[end-11:end,:cw] ; color=:goldenrod3, alpha=0.9, lw=3, markersize=8, markershape=:dtriangle, markerstrokecolor=:goldenrod3, label="univariate")
    plot!([0,1],[0,1],seriestype=:straightline,label=false,lw=3,color=:gray,linestyle=:dot)
    plot!(ylims=(-0.03,1.03))

    return mv_fig, cw_fig
end

function rc_guarantee( data, B ; markershape=:star, color=:turquoise, title="", size=size )
    fig = plot( ; legendfont, guidefont, titlefont, xlabel=L"\alpha", ylabel="expected loss", title, size  )

    plot!([0,0.51],[0,0.51],seriestype=:straightline,label="x=y",lw=3,color=:gray,linestyle=:dot)
    a =  multifilter( data, [:B], [(B,)] )
    plot!(a[:,:alpha], a[:,:guarantee] ; color, lw=3, label=false )
    scatter!( a[:,:alpha], a[:,:guarantee] ; color, markershape, markerstrokecolor=color, markersize=9, alpha=0.7, label=false )
    return fig
end

function uni_cw_vs( mv, cw, rc1, rcq, B ; size=size, title="", select=:, markersize=8, alpha=0.5, col=:vol_mean, ylabel=L"(set volume)$^{1/d}$"    )
    fig = plot( ; legendfont, guidefont, titlefont, xlabel="univariate coverage", ylabel, title, size  )

    c = [ :goldenrod3,  :slateblue, :turquoise, :darkslategray ]

    plot!(     cw[:,:cw], cw[:,col] ; color=c[1], lw=3, label=false )#"uni-conformal" )
    scatter!(  cw[:,:cw], cw[:,col] ; color=c[1], markerstrokecolor=c[1], markersize, alpha, markershape=:dtriangle, lw=3, label="uni-conformal" )

    plot!(     mv[:,:cw_lo], mv[:,col] ; color=c[2], lw=3, label=false) #"multi-conformal" )
    scatter!(  mv[:,:cw_lo], mv[:,col] ; color=c[2], markerstrokecolor=c[2], markersize, alpha=alpha, markershape=:circle, lw=3, label="multi-conformal" )

    cq = multifilter( rcq, [:B], [(B,)] )[select,:]
    plot!(     cq[:,:cw_lo], cq[:,col] ; color=c[4], lw=3, label=false)#"risk control, quantile" )
    scatter!(  cq[:,:cw_lo], cq[:,col] ; color=c[4], markerstrokecolor=c[4], markersize=markersize-1, alpha, markershape=:rect, lw=3, label="risk control, quantile" )

    c1 = multifilter( rc1, [:B], [(B,)] )[select,:]
    plot!(     c1[:,:cw_lo], c1[:,col] ; color=c[3], lw=3, label=false)#"risk control, 1 norm" )
    scatter!(  c1[:,:cw_lo], c1[:,col] ; color=c[3], markerstrokecolor=c[3], markersize=markersize+2, alpha, markershape=:star4, lw=3, label="risk control, 1 norm" )

    return fig
end



# read data
mv  = CSV.read(path*"/data/force_energy/mv_scaled.csv",DataFrame)
cw  = CSV.read(path*"/data/force_energy/cw_scaled.csv",DataFrame)
rc1 = CSV.read(path*"/data/force_energy/cr1_scaled.csv",DataFrame)
rcq = CSV.read(path*"/data/force_energy/crq_scaled.csv",DataFrame)

α_set=collect(0.4:0.05:0.95)

# guarantee verification
mv_fig, cw_fig = multiconformal_guarantees( α_set, mv, cw ; titles=[ "(b) full vector coverage", "(a) component-wise coverage"  ] )
rc1_fig        = rc_guarantee( rcq, 0.95 ; color=:darkslategray, markershape=:square, title="(b)  risk control, quantile loss" )
rcq_fig        = rc_guarantee( rc1, 0.95 ; markershape=:star4, color=:turquoise,      title="(a)  risk control, 1 norm loss" )
savefig( mv_fig,  string( "figures/force_energy/", prefix, "energy_force_multivariate_cov", suffix, extension ))
savefig( cw_fig,  string( "figures/force_energy/", prefix, "energy_force_univariate_cov",   suffix, extension ))
savefig( rc1_fig, string( "figures/force_energy/", prefix, "energy_force_rc1_guarantee",    suffix, extension ))
savefig( rcq_fig, string( "figures/force_energy/", prefix, "energy_force_rcq_guarantee",    suffix, extension ))

# comparison of coverage and volume
fig_vol    = uni_cw_vs( mv, cw[8:end,:], rc1, rcq, 0.95 ; col=:vol_mean, ylabel=L"(set volume)$^{1/d}$", title="(a)  volume comparison" )
fig_cov    = uni_cw_vs( mv, cw[8:end,:], rc1, rcq, 0.95 ; col=:mv,       ylabel="multivariate coverage", title="(b)  multi coverage comparison" )
fig_vol_08 = uni_cw_vs( mv, cw[8:end,:], rc1, rcq, 0.8  ; col=:vol_mean, ylabel=L"(set volume)$^{1/d}$", title="(a)  volume comparison" )
fig_cov_08 = uni_cw_vs( mv, cw[8:end,:], rc1, rcq, 0.8  ; col=:mv,       ylabel="multivariate coverage", title="(b)  multi coverage comparison" )
savefig( fig_vol,    string( "figures/force_energy/", prefix, "energy_force_unicov_vol",         suffix, extension ))
savefig( fig_cov,    string( "figures/force_energy/", prefix, "energy_force_unicov_multicov",    suffix, extension ))
savefig( fig_vol_08, string( "figures/force_energy/", prefix, "energy_force_unicov_vol_08",      suffix, extension ))
savefig( fig_cov_08, string( "figures/force_energy/", prefix, "energy_force_unicov_multicov_08", suffix, extension ))



