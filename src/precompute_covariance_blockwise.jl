
using LinearAlgebra
using Random
using CSV
using DataFrames
using Distributions
using Tullio
using SpecialFunctions
using NeighbourLists
using StaticArrays

# import Python libraries
using PyCall
ase   = pyimport("ase.io")
atoms = pyimport("ase")

# import local files
path  = @__DIR__
path  = path[1:end-4]
include( path*"/src/modules/auxiliary.jl")
include( path*"/src/modules/material_settings.jl")
include( path*"/src/modules/ellipsoid.jl")
include( path*"/src/modules/hypercube.jl")
include( path*"/src/modules/inner_kernel.jl")
include( path*"/src/modules/quip_descriptors.jl")

# ------------------------
# ==============================================================================
# prediction manager
# ==============================================================================
# -----------------------


# construct SOAP for atomic configurations
function get_descriptors( C ; type )
    files        = [ path_basic*"/data/configurations/GAP_silicon.xyz"]
    key          =   path_basic*"/data/configurations/key_structure.csv"
    quantities   = ["dft_energy"]
    derivatives  = ["dft_force"]
    stresses     = ["dft_virial"]
    task         =  "dia"
    inds         = [nothing]

    # run experiment
    xyz          = set_xyz( files, quantities, derivatives, key, inds, stresses  )
    descriptor   = set_SOAP( ; species=["Si"], r_cut=5.0 )
    settings     = set_settings( ;  descriptor )

    return describe( task, C ;  xyz, settings, type )[1]
end

# obtain sets of indices to parallelize covariance construction
function get_blocks(inds)
    patches = []
    for i in 1:ℓ(inds)
        for j in 1:ℓ(inds)
            push!( patches, (inds[i],inds[j]) )
        end
    end
    return patches
end

function get_blocks(inds_a,inds_b)
    patches = []
    for i in 1:ℓ(inds_a)
        for j in 1:ℓ(inds_b)
            push!( patches, (inds_a[i],inds_b[j]) )
        end
    end
    return patches
end


# obtain sets of indices to parallelize covariance construction for SYMMETRIC matrix
function get_blocks_symmetric(inds)
    patches = []
    for i in 1:ℓ(inds)
        push!( patches, (inds[i],inds[i]) )
        for j in i+1:ℓ(inds)
            push!( patches, (inds[i],inds[j]) )
        end
    end
    return patches
end

# given indices for a block of the covariance matrix, build and save that block
function build_blocks(loc_patches, path, cov_type)
    
    if cov_type in ["stress","stress_energy"]
        type="stress"
    else 
        type="force"
    end

    for (rows,cols) in loc_patches
        name = string( path, maximum(rows), "_", maximum(cols), "_K.csv" )  
        if C[rows]==C[cols]
            X  = get_descriptors( C[rows] ; type )
            rr = 1:ℓ(rows)
            K  = covariance(  X, rr, rr  ;  C=getfield(Main, Symbol(cov_type)) )
            CSV.write( name, Tables.table(K) )
        else
            X  = get_descriptors( C[[rows;cols]] ; type )
            rr = 1:ℓ(rows)
            cc = ℓ(rows).+(1:ℓ(cols))
            K  = covariance(  X, rr, cc  ;  C=getfield(Main, Symbol(cov_type)) )
            CSV.write( name, Tables.table(K) )
        end
    end
end

# remove covariance block files after the full matrix has been compiled
function remove_blocks(loc_patches,  path)
    for (rows,cols) in loc_patches
        name = string( path, maximum(rows), "_", maximum(cols), "_K.csv" )
        rm(name)
    end
end

# read and concatenate covariance blocks for a SYMMETRIC matrix
function compile_blocks_symmetric(inds, path)

    K = []
    mmm = 0
    for (i,rows) in enumerate( inds )
        RR = []

        for cols in inds[i:end]
            name = string( path, maximum(rows), "_", maximum(cols), "_K.csv" )
            push!( RR, Matrix(CSV.read(name,DataFrame)) )
        end

        if i==1
            push!( K, hcat(RR...) )
            mmm += size(RR[1],2)
        else
            n  = size(RR[1],1)
            push!( K, hcat(zeros(n,mmm),RR...) )
            mmm += size(RR[1],2)
        end
    end

    K = vcat(K...)
    CSV.write( string(path, "compiled_K.csv"), Tables.table(Symmetric(K)) )
end

# read and compile covariance blocks
function compile_blocks(inds, path)

    K = []
    mmm = 0
    for (i,rows) in enumerate( inds )
        RR = []

        for cols in inds
            name = string( path, maximum(rows), "_", maximum(cols),  "_K.csv" )
            push!( RR, Matrix(CSV.read(name,DataFrame)) )
        end
        push!( K, hcat(RR...) )
    end
    K = vcat(K...)
    CSV.write( string(path, "compiled_K.csv"), Tables.table(K) )

end

# read and compile covariance blocks
function compile_blocks(inds_a, inds_b, path)

    K = []
    mmm = 0
    for (i,rows) in enumerate( inds_a )
        RR = []

        for cols in inds_b
            name = string( path, maximum(rows), "_", maximum(cols),  "_K.csv" )
            push!( RR, Matrix(CSV.read(name,DataFrame)) )
        end
        push!( K, hcat(RR...) )
    end
    K = vcat(K...)
    CSV.write( string(path, "compiled_K.csv"), Tables.table(K) )

end


# subset of regressors approximation for covariance matrix
# needed:
# sor      = which molecules are our subset
# mol_sz   = how many force components per molecule
# inds     = how our molecules divided up in covariance patches 
function set_up_sor(sor, mol_sz, inds, path)
    
    # compute local indices
    psz       = length(inds[1])
    which_ind = div.(sor.-1,psz)  .+ 1
    local_ind = mod.(sor.-1,psz)  .+ 1

    # initialize
    Knm = zeros( sum(mol_sz), sum(mol_sz[sor]) ) 

    cols = 0
    for s in 1:length(sor)
        i    = which_ind[s]
        loc  = sum(mol_sz[inds[i]][1:local_ind[s]-1]) .+ (1:mol_sz[inds[i]][local_ind[s]])
        cols = cols[end] .+ (1:mol_sz[sor[s]]) 
    

        rows = 0
        for j in 1:length(inds)
            rows = rows[end] .+ (  1:sum(mol_sz[inds[j]])  )
            if j<i
                name           = string( path, maximum(inds[j]), "_", maximum(inds[i]),  "_K.csv" )
                Knm[rows,cols] = Matrix(CSV.read(name,DataFrame))[:,loc]
            else
                name           = string( path, maximum(inds[i]), "_", maximum(inds[j]), "_K.csv" )
                Knm[rows,cols] = transpose(Matrix(CSV.read(name,DataFrame))[loc,:])
            end
        end
    end
    CSV.write( string(path*"Knm_", length(sor), ".csv"), Tables.table(Knm))
end


# -----------------i-------
# ==============================================================================
# Implement tests
# ==============================================================================
# -----------------------

# parse arguments
if isempty(ARGS)
    println("Error: Please provide at least one argument.")
    exit(1)
end

loc_id     = parse(Int64, ARGS[1] ) 
num_tasks  = parse(Int64, ARGS[2] ) 
psz        = parse(Int64, ARGS[3] )
cov_type   = ARGS[4]
symmetric  = parse(Bool, lowercase(ARGS[5]))
SOR_approx = parse(Bool, lowercase(ARGS[6]))


# read configuration indices
path_basic = path
path       = string(  path, "/data/covariance/", cov_type, "_")
C          = CSV.read( path*"C.csv" ,DataFrame)[2:end,1]
if SOR_approx
    sor    = CSV.read( path*"sor.csv",DataFrame)[2:end,1] .- 1
    mol_sz = CSV.read( path*"mol_sz.csv",DataFrame)[2:end,1]
end


# divy up covariance blocks
inds        = split(1:ℓ(C),psz)
patches     = symmetric ? get_blocks_symmetric(inds) : get_blocks(inds)
loc_patches = patches[loc_id:num_tasks:end]

# build covariance blocks
build_blocks(loc_patches, path, cov_type)

# if proces 1, combine files
if loc_id==1 
    
    # read and concatenate blocks
    if symmetric
        if SOR_approx
            set_up_sor(sor, mol_sz, inds, path)
        else
            compile_blocks_symmetric(inds, path)
        end
    else
        if SOR_approx
            set_up_sor(sor, mol_sz, inds, path)
        else
            compile_blocks(inds, path)
        end
    end

    # clean up
    remove_blocks(patches, path)
end



