
# ------------------------
# ==============================================================================
# Set Modeling parameters
# ==============================================================================
# -----------------------


function set_SOAP(  ; r_cut=5., sigma=0.5, l_max=6, n_max=8, species=["Si"] )
    return ( r_cut=r_cut, sigma=sigma, l_max=l_max, n_max=n_max, species=species  )
end

# derivatives: ie: "dft_force"
function set_xyz( files, quantities, derivatives, key, inds, stresses ; tasks=["dia"] )
        
    # truth and unique indices for Silicon
    EC  = [ 153.28991, 56.25009, 56.25009, 0., 0., 72.17693 ]
    MOD = 88.5966967
    Si  = [ 1,2,6 ]
    sc  = 160.2176621 # GPa / (eV/Å³) 

    #tasks      = names(CSV.read(key, DataFrame))
    xyz        = Dict()
    xyz["key"] = key
    for (t,f,q,d,s,i) in zip(tasks, files, quantities, derivatives,stresses,inds)
        xyz[t] = (file=f, qoi=q, d=d, s=s, MOD=round.(MOD/sc; digits=6), EC=round.(EC/sc; digits=6), Si=Si, inds=i)
    end 
    return xyz
end

function set_settings( ; describe=describe, descriptor=nothing, forces=nothing )

    descriptor   = isnothing(descriptor)   ? set_SOAP()         : descriptor
    forces       = isnothing(forces)       ? descriptor.species : forces
    return (descriptor=descriptor, describe=describe, forces=forces )
end

