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
# Vacancy formation
# =======================================================================================
# ---------------------------------------------------------------------------------------

function alpha_coverage_vfe_bonf_M( conformal, α_set ; title="",ylims=(0,1.03), size=size, xlabel=L"$1-\alpha$", ylabel="coverage", rows=:  )
    cws = []
    cbs = []
    mvs = []
    mMs = []
    for a in α_set
        data_a = multifilter( conformal, [:alpha],[(a,)] )
        push!( cws, mean(data_a[rows,:cw_vfe_cov]) )
        push!( cbs, mean(data_a[rows,:cb_vfe_cov]) )
        push!( mvs, mean(data_a[rows,:mv_vfe_cov]) )
        push!( mMs, mean(data_a[rows,:mm_vfe_cov]) )
    end
    fig = plot( ; legendfont, guidefont, titlefont, xlabel, ylabel, ylims, title, size )
    plot!( α_set,  cws ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3,label=false)#"univariate" )
    plot!( α_set,  cbs ; color=:sienna,     markerstrokecolor=:sienna,     markershape=:+,         markersize=8,alpha=0.9, lw=3, label=false)#L"$\approx$univariate" )
    plot!( α_set,  mvs ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8,alpha=0.9, lw=3, label=false)#L"$\approx$multivariate" )
    plot!( α_set,  mMs ; color=:turquoise,  markerstrokecolor=:turquoise,  markershape=:star,      markersize=8,alpha=0.9, lw=3, label=false)#L"$\approx$multivariate" )

    scatter!( [], [] ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3, label="univariate" )
    scatter!( [], [] ; color=:sienna,     markerstrokecolor=:sienna,     markershape=:+,         markersize=8,alpha=0.9, lw=3, label="Bonferroni" )
    scatter!( [], [] ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8,alpha=0.9, lw=3, label="multivariate" )
    scatter!( [], [] ; color=:turquoise,  markerstrokecolor=:turquoise,  markershape=:star,      markersize=8,alpha=0.9, lw=3, label="prop-corrected" )

    plot!([0,1],[0,1],seriestype=:straightline,label=false,lw=3,color=:gray,linestyle=:dot)
    return fig
end


function alpha_ratio_ribbon_vfe( data, α_set, col_a, col_b, col_c  ; legend=:topright, ref=true, size=size, cb=:goldenrod3, cc=:sienna, xlabel=L"$1-\alpha$", ylabel="volume ratio", title="", rows=:, select=:alpha, quantiles=[0.25,0.5,0.75], l1=L"uni/$\approx$multi", l2=L"$\approx$uni/$\approx$multi"  )

    low, mid, high = Float64[], Float64[], Float64[]
    for a in α_set
        data_a = multifilter( data, [select],[(a,)] )
        ratio  = data_a[rows,col_a] ./ data_a[rows,col_b]
        qs     = quantile( ratio, quantiles )
        push!(  low,  qs[2]-qs[1] )
        push!(  mid,  qs[2] )
        push!( high, -qs[2]+qs[3] )
    end
    fig = plot( ; legendfont, guidefont, titlefont, xlabel, ylabel, title, size, legend )
    plot!( α_set,  mid,  ribbon=(low,high) ; color=cb, alpha=0.9, lw=3, label=l1, markershape=:dtriangle, markersize=8, markerstrokecolor=cb )

    low, mid, high = Float64[], Float64[], Float64[]
    for a in α_set
        data_a = multifilter( data, [select],[(a,)] )
        ratio  = data_a[rows,col_c] ./ data_a[rows,col_b]
        qs     = quantile( ratio, quantiles )
        push!(  low,  qs[2]-qs[1] )
        push!(  mid,  qs[2] )
        push!( high, -qs[2]+qs[3] )
    end
    plot!( α_set,  mid,  ribbon=(low,high) ; color=cc, alpha=0.9, lw=3, label=l2, markershape=:+, markersize=8, markerstrokecolor=cc  )
    if ref
        hline!( [1.] ; color=:gray, lw=3, linestyle=:dot,  label=false)
    end
    return fig
end

α_set=collect(0.4:0.05:0.95)

# read data
v_150 = CSV.read( path*"/data/vacancy_formation/vacancy_150_150_300_conformal.csv", DataFrame )
v_250 = CSV.read( path*"/data/vacancy_formation/vacancy_250_150_200_conformal.csv", DataFrame )
v_350 = CSV.read( path*"/data/vacancy_formation/vacancy_350_150_100_conformal.csv", DataFrame )

# coverage 
f_350 = alpha_coverage_vfe_bonf_M( v_350, α_set ;title="(a) vacancy formation energy" )
f_250 = alpha_coverage_vfe_bonf_M( v_250, α_set ;title="(a) vacancy formation energy" )
f_150 = alpha_coverage_vfe_bonf_M( v_150, α_set ;title="(a) vacancy formation energy" )
savefig( f_350, string( "figures/vacancy_formation/", prefix, "vacancy_multi",     suffix, extension ))
savefig( f_250, string( "figures/vacancy_formation/", prefix, "vacancy_multi_250", suffix, extension ))
savefig( f_150, string( "figures/vacancy_formation/", prefix, "vacancy_multi_150", suffix, extension ))

# volume
f_350    = alpha_ratio_ribbon_vfe( v_350, α_set, :cw_vfe_volume, :mm_vfe_volume, :cb_vfe_volume  ; title="(b) vacancy formation energy", l1="univariate", l2="Bonferroni" )
f_250    = alpha_ratio_ribbon_vfe( v_250, α_set, :cw_vfe_volume, :mm_vfe_volume, :cb_vfe_volume  ; title="(b) vacancy formation energy", l1="univariate", l2="Bonferroni" )
f_150    = alpha_ratio_ribbon_vfe( v_150, α_set, :cw_vfe_volume, :mm_vfe_volume, :cb_vfe_volume  ; title="(b) vacancy formation energy", l1="univariate", l2="Bonferroni" )
f_mv_150 = alpha_ratio_ribbon_vfe( v_150, α_set, :cw_vfe_volume, :mv_vfe_volume, :cb_vfe_volume  ; title="(b) vacancy formation energy", l1="univariate", l2="Bonferroni", legend=:false )
savefig( f_350,    string( "figures/vacancy_formation/", prefix, "vacancy_pc_ratio",     suffix, extension ))
savefig( f_250,    string( "figures/vacancy_formation/", prefix, "vacancy_pc_ratio_250", suffix, extension ))
savefig( f_150,    string( "figures/vacancy_formation/", prefix, "vacancy_pc_ratio_150", suffix, extension ))
savefig( f_mv_150, string( "figures/vacancy_formation/", prefix, "vacancy_mv_ratio_150", suffix, extension ))




