################################################################################
## LOAD ALL PACKAGES
using JuMP
using SumOfSquares
using MosekTools 
using DynamicPolynomials
using QuadGK
# using FastGaussQuadrature
using LinearAlgebra
import CairoMakie as cm # need to import due to conflict with SumOfSquares


################################################################################
## INTEGRATION OVER AN ELLIPSE
################################################################################
function integrate_monomial_ellipse(α, β)
    N(x) = (cos(x)^α) * (sin(x)^β)
    integral_val, _ = quadgk(N, 0, 2π)
    return ( 0.5^β / (α + β + 2)) * integral_val
end

function integrate_polynomial_ellipse(p)
    I = 0
    count = 0
    for exponent ∈ p.x.Z
        α = exponent[1]
        β = exponent[2]
        count += 1
        I += p.a[count] * integrate_monomial_ellipse(α, β)
    end
    return I
end



################################################################################
## SOLVE FOR AN ELLIPSE
################################################################################
function optimize_ellipse(d)
    # Extend the definition of u to include more elements
    # model = SOSModel(CSDP.Optimizer)
    model = SOSModel(Mosek.Optimizer)
    # Define the problem
    @polyvar x y
    MON = monomials([x, y], 0:d)
    @variable(model, u, Poly(MON))
    @variable(model, v, Poly(MON))
    Du = [differentiate(u, x) differentiate(u, y); differentiate(v, x) differentiate(v, y)]
    e = 0.5 * (Du + Du')
    div_u = Du[1, 1] + Du[2, 2]
    p = x^2 - y^2
    ∇p = differentiate(p, (x, y))
    ∇p_squared = (∇p[1])^2 + (∇p[2])^2
    int_divu = integrate_polynomial_ellipse(div_u)
    int_gradp = integrate_polynomial_ellipse(∇p_squared)
    μ = (∇p * ∇p') - 2 * e # defect measure, will be in PSD cone
    # K = @set 1 - x^2 - 4y^2 ≥ 0
    # @constraint(model, μ in PSDCone(), domain = K)
    dμ = d-1
    ω  = ceil(Int, dμ/2)
    m0 = monomials([x, y], 0:ω)
    m1 = monomials([x, y], 0:ω-1)
    
    r0 = 2*binomial(2+ω, ω)
    r1 = 2*binomial(2+ω-1, ω-1)
    
    @variable(model, Q0[1:r0, 1:r0], PSD)
    @variable(model, Q1[1:r1, 1:r1], PSD)

    I_0 = Matrix{Float64}(I, 2, 2)
    σ₀ = kron(I_0,  m0)' * Q0 *  kron(I_0,  m0)
    σ₁ = kron(I_0,  m1)' * Q1 *  kron(I_0,  m1)
    σ = σ₀ + (1 - x^2 - 4y^2) * σ₁

    @constraint(model, μ[1,1] == σ[1,1])
    @constraint(model, μ[2,1] == σ[2,1])
    @constraint(model, μ[2,2] == σ[2,2])
    @objective(model, Min, 0.5int_gradp - int_divu)
    optimize!(model)
    # Get the optimal value of the objective
    opt_val = objective_value(model)
    opt_time = solve_time(model) # Optimization time in seconds

    println("Optimal Value: ", opt_val)
    println("Optimization Time: ", opt_time, " seconds")
    # output something
    return (x, y), value.(u), value.(v), value.(μ),opt_val,opt_time
end

################################################################################
## PLOT WRINKLING PATTERNS
################################################################################
function get_wrinkles(xy, sol, tol)
    μ = sol[4]
    x = xy[1]
    y = xy[2]
    if  x^2 + 4y^2 > 1
        return (NaN, NaN)
    else
        μ_val = [μ[1, 1](x, y) μ[1, 2](x, y); μ[2, 1](x, y) μ[2, 2](x, y)]
        ev = eigen(μ_val)
        # From Tobasco et al, peaks and troughs are along eigenvector of μ
        # with "zero" eigenvalue
        if ev.values[1] < tol
            return (ev.vectors[1, 1], ev.vectors[2, 1])
        else
            return (NaN, NaN)
        end
    end
end

function plot_wrinkles(sol,x,y,tol)
    # remove the arrow heads to make a nicer plot
    f(x) = cm.Point2f( get_wrinkles(x, sol, tol) )
    fig, ax, pl = cm.streamplot(f, xp, yp, 
                                arrow_size=0, 
                                stepsize=1e-3, 
                                maxsteps=500)
    ax.title = "Tolerance = $tol and Degree = $d" 
    # Hide grid lines
    ax.xgridvisible = false
    ax.ygridvisible = false  
    
    # Add ellipse border (x^2 + 4y^2 = 1)
    θ = LinRange(0, 2π, 200)  # Angle values for the parametric equation
    x_ellipse = cos.(θ)       # x-coordinates of the ellipse
    y_ellipse = 0.5 * sin.(θ) # y-coordinates of the ellipse
    cm.lines!(ax, x_ellipse, y_ellipse, color=:black, linewidth=2)
    return fig, ax, pl
end

################################################################################
## RUN SOME TESTS
################################################################################
d = 10; # degree of polynomials 
tol =1e-2 # tolerance for small eigenvalues
sol = optimize_ellipse(d)
opt_val = sol[5]
opt_time = sol[6]
xp = LinRange(-1, 1, 500) 
yp = LinRange(-1, 1, 500) 
plt = plot_wrinkles(sol, xp, yp, tol)
plt[1] # show plot