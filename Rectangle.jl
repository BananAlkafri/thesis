
################################################################################
## LOAD ALL PACKAGES
################################################################################
using JuMP
using SumOfSquares
using MosekTools
# using CSDP 
using DynamicPolynomials
using QuadGK
# using FastGaussQuadrature
using LinearAlgebra
import CairoMakie as cm # need to import due to conflict with SumOfSquares


################################################################################
## INTEGRATION OVER A RECTANGLE
################################################################################

function integrate_monomial_rectangle(α, β)
    return (1 - (-1)^(α+1)) / (α + 1) * (2^(β + 1) - (-2)^(β + 1)) / (β + 1)
end


function integrate_polynomial_rectangle(p)
    I = 0
    count = 0
    for exponent ∈ p.x.Z
        α = exponent[1]
        β = exponent[2]
        count += 1
        I += p.a[count] * integrate_monomial_rectangle(α, β)
    end
    return I
end



################################################################################
## SOLVE FOR A CIRCLE
################################################################################
function optimize_rectangle(d)
    # Extend the definition of u to include more elements
    model = SOSModel(Mosek.Optimizer)
    # Define the problem
    @polyvar x y
    MON = monomials([x, y], 0:d)
    @variable(model, u, Poly(MON))
    @variable(model, v, Poly(MON))
    Du = [differentiate(u, x) differentiate(u, y); differentiate(v, x) differentiate(v, y)]
    e = 0.5 * (Du + Du')
    div_u = Du[1, 1] + Du[2, 2]
    p = x^2 + y^2
    ∇p = differentiate(p, (x, y))
    ∇p_squared = (∇p[1])^2 + (∇p[2])^2
    int_divu = integrate_polynomial_rectangle(div_u)
    int_gradp = integrate_polynomial_rectangle(∇p_squared)
    μ = (∇p * ∇p') - 2 * e # defect measure, will be in PSD cone
  
    
    dμ = d-1
    ω  = ceil(Int, dμ/2)
    m0 = monomials([x, y], 0:ω)
    m1 = monomials([x, y], 0:ω-1)
    
    r0 = 2*binomial(2+ω, ω)
    r1 = 2*binomial(2+ω-1, ω-1)
    
    @variable(model, Q0[1:r0, 1:r0], PSD)
    @variable(model, Q1[1:r1, 1:r1], PSD)
    @variable(model, Q2[1:r1, 1:r1], PSD)

    I_0 = Matrix{Float64}(I, 2, 2)

    σ₀ = kron(I_0,  m0)' * Q0 *  kron(I_0,  m0)
    σ₁ = kron(I_0,  m1)' * Q1 *  kron(I_0,  m1)
    σ₂ = kron(I_0,  m1)' * Q2 *  kron(I_0,  m1)


    σ = σ₀ + (1 - x^2) * σ₁ + (4 - y^2) * σ₂


    @constraint(model, μ[1,1] == σ[1,1])
    @constraint(model, μ[2,1] == σ[2,1])
    @constraint(model, μ[2,2] == σ[2,2])

    @objective(model, Min, 0.5int_gradp - int_divu)
    optimize!(model)
    println(value.(μ))
    # output something
    return (x, y), value.(u), value.(v), value.(μ)
    
end



################################################################################
## PLOT WRINKLING PATTERNS
################################################################################
function get_wrinkles(xy, sol, tol)
    μ  = sol[4]
    x = xy[1]
    y = xy[2]
    μ_val = [ μ[1,1](x,y) μ[1,2](x,y); μ[2,1](x,y) μ[2,2](x,y) ]
    ev = eigen(μ_val)
    # From Tobasco et al, peaks and troughs are along eigenvector of μ
    # with "zero" eigenvalue
    if ev.values[1] < tol
        return (ev.vectors[1, 1], ev.vectors[2, 1])
    else
        return (NaN, NaN)
    end
    return (ev.vectors[1, 1], ev.vectors[2, 1]) 
end

function plot_wrinkles(sol, xp, yp, tol, d)
    # Define the function for calculating wrinkle directions
    f(xy) = cm.Point2f(get_wrinkles(xy, sol, tol))
    
    # Create a figure and axis for the plot
    fig = cm.Figure(resolution = (600, 450))
    ax = cm.Axis(fig[1, 1], title = "Tolerance = $tol and Degree = $d", xlabel = "x", ylabel = "y")
    
    # Hide grid lines
    ax.xgridvisible = false
    ax.ygridvisible = false
    
    # Generate the wrinkle streamplot within the domain
    cm.streamplot!(ax, f, xp, yp, arrow_size=0, stepsize=5e-3, maxsteps=500)

    # Draw a rectangular boundary with [-1, 1] in x and [-2, 2] in y
    cm.lines!(ax, [-1, 1, 1, -1, -1], [-2, -2, 2, 2, -2], color=:black, linewidth=2)
      
    return fig, ax
end



################################################################################
## RUN SOME TESTS
################################################################################
d   = 20; # degree of polynomials 
tol = 1e-2 # tolerance for small eigenvalues
sol = optimize_rectangle(d)
xp = LinRange(-1, 1, 500) 
yp = LinRange(-2, 2, 500) 
plt = plot_wrinkles(sol, xp, yp, tol, d)
plt[1] # show plot

