

log_V_unit_sphere(d)     = log(2/d) +  (d/2)log(π) - log( gamma(d/2))
log_V_ellipsoid(d,r²,Σ)  = log_V_unit_sphere(d) + (d/2)*log(r²) + (1/2)*logabsdet(Σ)[1]
V_ellipsoid(d,r²,Σ)      = exp(  log_V_unit_sphere(d) + (d/2)*log(r²) + (1/2)*logabsdet(Σ)[1]  )
R(θ)                     = [cos(θ) sin(θ); -sin(θ) cos(θ)]


function get_ellipse_points(μx, μy, rx, ry, θ)
    ϕ       = range(0, 2*pi, length=100)
    ellipse = [rx*cos.(ϕ)   ry*sin.(ϕ)] * R(θ)
    return μx .+ ellipse[:,1],   μy .+ ellipse[:,2]
end


function project_ellipse(Σ, r²)
    basis = eigen(Σ).vectors[:,end-1:end]
    ev    = eigen( basis' * Σ * basis  )
    #ev    = eigen( Σ[3:end,3:end]  )

    θ  = atan( ev.vectors[2,2] / ev.vectors[1,2] )
    θ += θ < 0 ? 2π : 0

    return get_ellipse_points( 0., 0., sqrt( r² * ev.values[2] ), sqrt( r² * ev.values[1] ), θ)
end


function distance2boundary( μ, Σ, y, r² )
    d₁ = (y-μ)' * ( Σ \ (y-μ))
    return d₁ - r²
end

function componentwise_coverage( μ, Σ, y, r²  )
    s = zeros(length(μ))
    n = 0
    for i in 1:length(μ)
        d    = diag(inv(Σ[i])) 
        s[i] = sum(   d.*(y[i]-μ[i]).^2   .<   r²   )
        n    += length(μ[i])
    end
    return sum(s)/n
end

function componentwise_coverage_upper( μ, Σ, y, r²  )
    s = zeros(length(μ))
    n = 0
    for i in 1:length(μ)
        s[i] = sum(   (y[i]-μ[i]).^2 ./ diag(Σ[i])   .<   r²   )
        n    += length(μ[i])
    end
    return sum(s)/n
end

function err_componentwise_coverage( ε, Σ, r²  )
    s = zeros(length(ε))
    n = 0
    for i in 1:length(ε)
        d    = diag(inv(Σ[i]))
        n   += length(ε[i])
        s[i] = sum( (d.*ε[i].^2) .< r² )
    end
    return sum(s)/n
end

function err_componentwise_coverage_upper( ε, Σ, r²  )
    s = zeros(length(ε))
    n = 0
    for i in 1:length(ε)
        n   += length(ε[i])
        s[i] = sum( ( ε[i].^2 ./ diag(Σ[i]) )  .< r² )
    end
    return sum(s)/n
end

function set_coverage( μ, Σ, y, r² ; j=1e-8 )
    s = [ dot(t-m, H(C;j)  \(t-m) )    <= r²   for (t,m,C) in zip(y,μ,Σ) ] 
    return  mean(s)
end


function energy2elastic_ellipse( μ, Σ, r², V ; ε, steps ) 
    new_μ, new_Σ, vmean, vstd  = [], [], [], []
    n  = length(steps) 
    for (h,a,b,c) in zip( steps, nest(μ,n), nest(Σ,n), nest(V,n) ) 
        temp = [] 
        for (m,C,v̄) in zip(a,b,c) 
            M  = elastic_constant_operator(ε,h,v̄)
            Cn = M*C*M'
            push!( new_μ,  M*m )
            push!( new_Σ,  Cn  )
            try 
                push!( temp, V_ellipsoid(6,r²,H(Cn;j=0.)) )
            catch e
                println(e)
            end
         end
         #push!( vmean, mean(temp) )
         push!( vmean, median(temp) )
         push!( vstd,  std(temp)  )
    end
    return new_μ, new_Σ, vmean, vstd
end



# We use 24 perturbations to the original configuration to compute energy
# operator is 6 x 24
# order:  [ 11: ++ +- -+ -- ] [22] [33] [12] [13] [23]
function elastic_constant_operator(ε,h,V)
    M = zeros(6,24)
    for i in 1:6
        M[i, 4(i-1)+1:4i] = [1 -1 -1  1]
    end
    return M./(4ε*h*V')
end


function ellipsoid(center, coefs, ngrid=25)
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1 ./ sqrt.(coefs)

    # Set of all spherical angles:
    u = range(0, 2pi, length=ngrid)
    v = range(0, pi, length=ngrid)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = [rx * x * y for (x, y) in  Iterators.product(cos.(u), sin.(v))]
    y = [ry * x * y for (x, y) in Iterators.product(sin.(u), sin.(v))]
    z = [rz * x * y for (x, y) in Iterators.product(ones(length(u)), cos.(v))]
    return x .+ center[1], y .+ center[2], z .+ center[3]
end

function plot_ellipsoid( M )
    F = svd(M)
    U = F.U
    V = F.V
    C = [0 0 0]

    x, y, z = ellipsoid(C, F.S)
    s =surface(x, y, z, aspect_ratio=:equal)

    return s
end
