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
# Elastic constant
# =======================================================================================
# ---------------------------------------------------------------------------------------

function alpha_coverage( conformal, α_set ; legend, size=size, title="",ylims=(0,1), select=:cw_alpha, mv=:mv_fullcov, cw=:cw_fullcov,norm=1, xlabel=L"$1-\alpha$", ylabel="multivariate coverage", rows=1:100  )
    cws = []
    mvs = []
    for a in α_set
        data_a = multifilter( conformal, [select],[(a,)] )
        push!( cws, mean(data_a[rows,cw])/norm )
        push!( mvs, mean(data_a[rows,mv])/norm )
    end
    fig = plot( ; legendfont, guidefont, titlefont, xlabel, ylabel, ylims, title, size, legend )
    plot!( α_set,  cws ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3, label=false)#"univariate" )
    plot!( α_set,  mvs ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8, alpha=0.9, lw=3, label=false)#"multivariate" )

    scatter!( [], [] ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3, label="univariate" )
    scatter!( [], [] ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8, alpha=0.9, lw=3, label="multivariate" )


    plot!([0,1],[0,1],seriestype=:straightline,label=false,lw=3,color=:gray,linestyle=:dot)
    return fig
end

function alpha_coverage_bonf( conformal, bonf, α_set ; size=size, title="",ylims=(0,1.03), legend, select=:cw_alpha, mv=:mv_fullcov, cw=:cw_fullcov,norm=1, xlabel=L"$1-\alpha$", ylabel="multivariate coverage", rows=1:100  )
    cws = []
    mvs = []
    bcw = []
    for a in α_set
        data_a = multifilter( conformal, [select],[(a,)] )
        data_b = multifilter( bonf,      [select],[(a,)] )
        push!( cws, mean(data_a[rows,cw])/norm )
        push!( mvs, mean(data_a[rows,mv])/norm )
        push!( bcw, mean(data_b[rows,cw])/norm )
    end
    fig = plot( ; legendfont, guidefont, titlefont, size, xlabel, ylabel, ylims, title, legend )
    plot!( α_set,  cws ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3,label=false )
    plot!( α_set,  bcw ; color=:sienna,     markerstrokecolor=:sienna,     markershape=:+,         markersize=8, alpha=0.9, lw=3,label=false )
    plot!( α_set,  mvs ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8, alpha=0.9, lw=3,label=false )

    scatter!( [], [] ; color=:goldenrod3, markerstrokecolor=:goldenrod3, markershape=:dtriangle, markersize=8, alpha=0.9, lw=3, label="univariate" )
    scatter!( [], [] ; color=:sienna,     markerstrokecolor=:sienna,     markershape=:+,         markersize=8, alpha=0.9, lw=3, label="Bonferroni" )
    scatter!( [], [] ; color=:slateblue,  markerstrokecolor=:slateblue,  markershape=:x,         markersize=8, alpha=0.9, lw=3, label="multivariate" )

    plot!([0,1],[0,1],seriestype=:straightline,label=false,lw=3,color=:gray,linestyle=:dot)
    return fig
end

function alpha_ratio_ribbon( data, α_set, col_a, col_b  ; ref=true, size=size, color=:goldenrod3, xlabel=L"$1-\alpha$", ylabel="volume ratio", title="", rows=1:100, select=:cw_alpha, quantiles=[0.25,0.5,0.75]  )

    low, mid, high = Float64[], Float64[], Float64[]
    for a in α_set
        data_a = multifilter( data, [select],[(a,)] )
        ratio  = data_a[rows,col_a] ./ data_a[rows,col_b]
        qs     = quantile( ratio, quantiles )
        push!(  low,  qs[2]-qs[1] )
        push!(  mid,  qs[2] )
        push!( high, -qs[2]+qs[3] )
    end
    fig = plot( ; legendfont, guidefont, titlefont, xlabel, ylabel, title, size )
    plot!( α_set,  mid,  ribbon=(low,high) ; color, markerstrokecolor=color, markersize=8, markershape=:x,  alpha=0.9, lw=3, label=false )
    if ref
        hline!( [1.] ; color=:gray, lw=3, linestyle=:dash,  label=false)
    end
    return fig
end

function alpha_ratio_ribbon_bonf( data, bonf, α_set, col_a, col_b  ; ref=true, size=size, color=:goldenrod3, colorb=:sienna, xlabel=L"$1-\alpha$", ylabel="volume ratio", title="", rows=1:100, select=:cw_alpha, quantiles=[0.25,0.5,0.75]  )

    low,   mid,  high = Float64[], Float64[], Float64[]
    blow, bmid, bhigh = Float64[], Float64[], Float64[]
    for a in α_set
        data_a = multifilter( data, [select],[(a,)] )
        data_b = multifilter( bonf, [select],[(a,)] )
        ratio  = data_a[rows,col_a] ./ data_a[rows,col_b]
        qs     = quantile( ratio, quantiles )
        push!(  low,  qs[2]-qs[1] )
        push!(  mid,  qs[2] )
        push!( high, -qs[2]+qs[3] )

        ratio  = data_b[rows,col_a] ./ data_a[rows,col_b]
        qs     = quantile( ratio, quantiles )
        push!(  blow,  qs[2]-qs[1] )
        push!(  bmid,  qs[2] )
        push!( bhigh, -qs[2]+qs[3] )
    end
    fig = plot( ; legendfont, guidefont, titlefont, xlabel, ylabel, title, size )
    plot!( α_set,  mid,  ribbon=(low,high) ; color, markerstrokecolor=color, markersize=8, markershape=:dtriangle, alpha=0.9, lw=3, label=false )
    plot!( α_set,  bmid,  ribbon=(blow,bhigh) ; color=colorb, markerstrokecolor=colorb, markersize=8, markershape=:+, alpha=0.9, lw=3, label=false )

    if ref
        hline!( [1.] ; color=:gray, lw=3, linestyle=:dot,  label=false)
    end
    return fig
end

α_set=collect(0.4:0.05:0.95)

# read data
e_16    = CSV.read(path*"/data/energy_elastic_constant/ec_energy_config329_05_05_nC350_nT100_conformal.csv",DataFrame)
e_16_b  = CSV.read(path*"/data/energy_elastic_constant/ec_energy_config329_05_05_nC350_nT100_bonf.csv",DataFrame)
s_16    = CSV.read(path*"/data/stress_elastic_constant/ec_stress_config329_05_nC15_nT175_conformal.csv", DataFrame)
s_16_b  = CSV.read(path*"/data/stress_elastic_constant/ec_stress_config329_05_nC15_nT175_bonf.csv", DataFrame)
e_16_01 = CSV.read(path*"/data/energy_elastic_constant/ec_energy_config329_01_01_nC350_nT100_conformal.csv",DataFrame)
e_54    = CSV.read(path*"/data/energy_elastic_constant/ec_energy_config549_05_05_nC350_nT100_conformal.csv",DataFrame)

# coverage
r_mv_e16 = alpha_coverage_bonf( e_16, e_16_b, α_set ; title="(a) elastic constant from energy", mv=:mv_redcov, cw=:cw_redcov, norm=1, ylabel="multivariate coverage", legend=false )
r_mv_s16 = alpha_coverage_bonf( s_16, s_16_b, α_set ; title="(b) elastic constant from stress", mv=:mv_redcov, cw=:cw_redcov, norm=1, ylabel="multivariate coverage", legend=:bottomright )
f_mv_e16 = alpha_coverage_bonf( e_16, e_16_b, α_set ; title="(a) elastic constant from energy", mv=:mv_fullcov, cw=:cw_fullcov, norm=1, ylabel="multivariate coverage", legend=:topleft )
f_mv_s16 = alpha_coverage_bonf( s_16, s_16_b, α_set ; title="(b) elastic constant from stress", mv=:mv_fullcov, cw=:cw_fullcov, norm=1, ylabel="multivariate coverage", legend=false )
r_cw_e16 = alpha_coverage_bonf( e_16, e_16_b, α_set ; title="(a) elastic constant from energy", mv=:mv_cwredcov_upper, cw=:cw_cwredcov, norm=3, ylabel="univariate coverage", legend=:bottomright )
r_cw_s16 = alpha_coverage_bonf( s_16, s_16_b, α_set ; title="(b) elastic constant from stress", mv=:cw_redcov_upper, cw=:cw_cwredcov, norm=3, ylabel="univariate coverage", legend=false )
savefig( r_mv_e16, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redcov_bonf", suffix, extension ))
savefig( r_mv_s16, string( "figures/stress_elastic_constant/", prefix, "stress_ec_redcov_bonf", suffix, extension ))
savefig( f_mv_e16, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullcov_bonf", suffix, extension ))
savefig( f_mv_s16, string( "figures/stress_elastic_constant/", prefix, "stress_ec_fullcov_bonf", suffix, extension ))
savefig( r_cw_e16, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redcov_unicov_bonf", suffix, extension ))
savefig( r_cw_s16, string( "figures/stress_elastic_constant/", prefix, "stress_ec_redcov_unicov_bonf", suffix, extension ))

# step size 0.01
r_01 = alpha_coverage( e_16_01, α_set ; title="(a) three dimensional target", mv=:mv_redcov, cw=:cw_redcov, norm=1, ylabel="multivariate coverage", legend=:topleft )
f_01 = alpha_coverage( e_16_01, α_set ; title="(b) six dimensional target", mv=:mv_fullcov, cw=:cw_fullcov, norm=1, ylabel="multivariate coverage", legend=false )
savefig( r_01, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redcov_step01", suffix, extension ))
savefig( f_01, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullcov_step01", suffix, extension ))

# Si54
r_54 = alpha_coverage( e_54, α_set ; title="(a) three dimensional target", mv=:mv_redcov, cw=:cw_redcov, norm=1, ylabel="multivariate coverage", legend=:topleft )
f_54 = alpha_coverage( e_54, α_set ; title="(b) six dimensional target", mv=:mv_fullcov, cw=:cw_fullcov, norm=1, ylabel="multivariate coverage", legend=false )
savefig( r_54, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redcov_config549", suffix, extension ))
savefig( f_54, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullcov_config549", suffix, extension ))

# volume
quantiles = [0.25,0.5,0.75]
r_mv_e16 = alpha_ratio_ribbon_bonf( e_16, e_16_b, α_set, :cw_ec_volume_red, :mv_ec_volume_red ; title="elastic constant from energy", rows=1:100, quantiles  )
r_mv_s16 = alpha_ratio_ribbon_bonf( s_16, s_16_b, α_set, :cw_ec_volume_red, :mv_ec_volume_red ; title="(c) elastic constant from stress", rows=1:100, quantiles  )
f_mv_e16 = alpha_ratio_ribbon_bonf( e_16, e_16_b, α_set, :cw_ec_volume,     :mv_ec_volume     ; title="(c) elastic constant from energy", rows=1:100, quantiles  )
f_mv_s16 = alpha_ratio_ribbon_bonf( s_16, s_16_b, α_set, :cw_ec_volume,     :mv_ec_volume     ; title="(d) elastic constant from stress", rows=1:100, quantiles  )
savefig( r_mv_e16, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redvol_bonf", suffix, extension ))
savefig( r_mv_s16, string( "figures/stress_elastic_constant/", prefix, "stress_ec_redvol_bonf", suffix, extension ))
savefig( f_mv_e16, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullvol_bonf", suffix, extension ))
savefig( f_mv_s16, string( "figures/stress_elastic_constant/", prefix, "stress_ec_fullvol_bonf", suffix, extension ))

# step size 0.01
r_01 = alpha_ratio_ribbon( e_16_01, α_set, :cw_ec_volume_red, :mv_ec_volume_red ; title="(c) three dimensional target", rows=1:100, quantiles  )
f_01 = alpha_ratio_ribbon( e_16_01, α_set, :cw_ec_volume,     :mv_ec_volume     ; title="(d) six dimensional target", rows=1:100, quantiles  )
savefig( r_01, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redvol_step01", suffix, extension ))
savefig( f_01, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullvol_step01", suffix, extension ))

# step size 0.01
r_54 = alpha_ratio_ribbon( e_54, α_set, :cw_ec_volume_red, :mv_ec_volume_red ; title="(c) three dimensional target", rows=1:100, quantiles  )
f_54 = alpha_ratio_ribbon( e_54, α_set, :cw_ec_volume,     :mv_ec_volume     ; title="(d) six dimensional target", rows=1:100, quantiles  )
savefig( r_54, string( "figures/energy_elastic_constant/", prefix, "energy_ec_redvol_config549", suffix, extension ))
savefig( f_54, string( "figures/energy_elastic_constant/", prefix, "energy_ec_fullvol_config549", suffix, extension ))







