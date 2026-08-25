

# assume a centered N cube defined by the intervals (-wⱼ, wⱼ) for j in 1:N
# Let V be a vector pointing out from the center of the N cube
# find a>0 such that aV lies on the surface of the hypercube
function vector_intersect( w, V )

    plus  = V .> 0.
    minus = V .< 0.

    upper = [ w[plus]./V[plus]; -w[minus]./V[minus]  ]   
    return minimum(upper)
end

# consider an N cube with sides given by the intervals (μⱼ-wⱼ, μⱼ+wⱼ ) for j in 1:N
# let d₁ = ||(y-μ)/σ||²
# let d₂ = ||(s-μ)/σ||² where s is the point where μ+a*(y-μ) intersects the hypercube for positive a
# return d₁-d₂ = (1-a²)d₁  
function distance2boundary( μ, w, y, σ ; loss_only=true, stdev=true )  # TODO use this 
    d₁ = stdev ? sum(( (y-μ)./σ ).^2) : sum(( (y-μ) ).^2)
    a  = vector_intersect( w, y-μ )  # scale vectors by σ here?
    loss_only && relu( (1-a^2)*d₁  )
    return (1-a^2)*d₁
end

function distance2boundary( μ, w, y, σ, nSet )
    m = floor( length(μ) / nSet )
    d = distance2boundary.( split(μ[1:r(m*nSet)],nSet), split(w[1:r(m*nSet)],nSet), split(y[1:r(m*nSet)],nSet), split(σ[1:r(m*nSet)],nSet) )
    return mean(d)
end


# input 24N cube given by the intervals (μⱼ-wⱼ, μⱼ+wⱼ ) for j in 1:24N
# output 6N cube with components corresponding to elastic constant, volume of each of N 6cubes
#=
function energy2elastic_cube( μ, w, V ; ε, steps )   
    μ = reshape(μ,4,6,:) ./ (4ε * reshape(V,4,6,:))
    w = reshape(w,4,6,:) ./ (4ε * reshape(V,4,6,:))
    s = [1,-1,-1,1]
    n = length(steps)

    maxs = reshape(sum( s.*μ + w ; dims=1), 6,n,:) ./ steps'
    mins = reshape(sum( s.*μ - w ; dims=1), 6,n,:) ./ steps'
    v    = permutedims( prod( maxs-mins ; dims=1 ), [2,3,1])

    return reshape(mins,:,1), reshape(maxs,:,1), mean(v; dims=2), std(v; dims=2)
end
=#
function energy2elastic_cube( μ, w, V ; ε, steps )
    
    s   = [1,-1,-1,1]
    n   = length(steps)
    ec  = reshape( sum( reshape(μ,4,6,:) .* s ./ (4ε * reshape(V,4,6,:))   ; dims=1 ), 6, n, :) ./ steps'
    err = reshape( sum( reshape(w,4,6,:)      ./ (4ε * reshape(V,4,6,:))   ; dims=1 ), 6, n, :) ./ steps' 
    v   = permutedims( prod( 2*err ; dims=1 ), [2,3,1])

    return reshape(ec-err,:,1), reshape(ec+err,:,1), median(v; dims=2), std(v; dims=2)
    #return reshape(ec-err,:,1), reshape(ec+err,:,1), mean(v; dims=2), std(v; dims=2)
end


function componentwise_coverage( μ, w, y ) 
    upper = y .< μ+w
    lower = y .> μ-w
    return sum(upper.*lower)/length(y)
end 

function set_coverage( μ, w, y ; nSet )  #TODO ensure input dimensions will be divisible by nSet
    m = floor( length(μ) / nSet ) 
    μ = reshape(μ[1:r(m*nSet)],nSet,:)
    w = reshape(w[1:r(m*nSet)],nSet,:)
    y = reshape(y[1:r(m*nSet)],nSet,:)

    upper = prod( y .< μ+w ; dims=1 )
    lower = prod( y .> μ-w ; dims=1 )
    
    return sum( upper.*lower ) / size(y,2)
end


function paired_set_coverage( μa, μb, wa, wb, ya, yb )
    m = minimum([length(μa),length(μb)])
    μ = [μa[1:m]' ; μb[1:m]']
    w = [wa[1:m]' ; wb[1:m]']
    y = [ya[1:m]' ; yb[1:m]']

    return mean(prod( abs.(y) .< μ+w ; dims=1 ))
end


function plot_rect_prism( xl, yl, zl ; color )
    
    # Define the 8 vertices of the prism
    vertices = [
        [-xl, -yl, -zl], [xl, -yl, -zl], [xl, yl, -zl], [-xl, yl, -zl],
        [-xl, yl,  zl],  [xl, yl,  zl], [xl, -yl,  zl], [xl, yl,  zl],
    ]
    
    # Define the 12 edges as pairs of vertex indices
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 1), # bottom face
        (5, 6), (6, 7), (7, 8), (8, 5), # top face
        (1, 5), (2, 6), (3, 7), (4, 8)  # vertical edges
    ]
    
    # Extract x, y, z coordinates for plotting
    x_coords = [v[1] for v in vertices]
    y_coords = [v[2] for v in vertices]
    z_coords = [v[3] for v in vertices]
    
    # Plot the prism
    p3 = plot3d(x_coords, y_coords, z_coords,
                line = (:path, 2), # set line style and width
                legend = false; color)
   
    vertices = [
        [-xl, -yl, -zl], [xl, -yl, -zl], [xl, yl, -zl], [-xl, yl, -zl],
        [-xl, -yl, zl],  [xl, -yl, zl],  [xl, yl, zl],  [-xl, yl, zl]
    ]


    # Add the edges
    for edge in edges
        i, j = edge
        plot3d!([vertices[i][1], vertices[j][1]],
               [vertices[i][2], vertices[j][2]],
               [vertices[i][3], vertices[j][3]],
               line = (:path, 2); color)
    end

    return p3
end


