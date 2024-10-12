################################################################################
## LOAD ALL PACKAGES
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
## INTEGRATION OVER A SQUARE
################################################################################
#calculate the integral of a polynomial on a square domain [-1,1]x[-1,1]
function integrate_monomial_square(α, β)
    return (1 + (-1)^α) / (α + 1) * (1 + (-1)^β) / (β + 1)
end

#integrate bivariate polynomial on a square
function integrate_polynomial_square(p)
    I = 0
    count = 0
    for exponent ∈ p.x.Z
        α = exponent[1]
        β = exponent[2]
        count += 1
        I += p.a[count] * integrate_monomial_square(α, β)
    end
    return I
end


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
## SOLVE FOR A SQUARE
################################################################################
function optimize_square(d)
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
    int_divu = integrate_polynomial_square(div_u)
    int_gradp = integrate_polynomial_square(∇p_squared)
    μ = (∇p * ∇p') - 2 * e # defect measure, will be in PSD cone
    K = @set 1 - x^2 ≥ 0 && 1 - y^2 ≥ 0
    @constraint(model, μ in PSDCone(), domain = K)
    @objective(model, Min, 0.5int_gradp - int_divu)
    optimize!(model)
    # output something
    return (x, y), value.(u), value.(v), value.(μ)
end


################################################################################
## SOLVE FOR A SQUARE
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
    p = x^2 + y^2
    ∇p = differentiate(p, (x, y))
    ∇p_squared = (∇p[1])^2 + (∇p[2])^2
    int_divu = integrate_polynomial_ellipse(div_u)
    int_gradp = integrate_polynomial_ellipse(∇p_squared)
    μ = (∇p * ∇p') - 2 * e # defect measure, will be in PSD cone
    K = @set 1 - x^2 - 4y^2 ≥ 0
    @constraint(model, μ in PSDCone(), domain = K)
    @objective(model, Min, 0.5int_gradp - int_divu)
    optimize!(model)
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
end

function plot_wrinkles(sol,x,y,tol, d)
    # remove the arrow heads to make a nicer plot
    f(x) = cm.Point2f( get_wrinkles(x, sol, tol) )
    fig, ax, pl = cm.streamplot(f, xp, yp, 
                                arrow_size=0, 
                                stepsize=5e-3, 
                                maxsteps=500)
    ax.title = "Tolerance = $tol and Degree = $d"
    ax.xlabel = "x"
    ax.ylabel = "y"                         
    return fig, ax, pl
end

################################################################################
## RUN SOME TESTS
################################################################################
d   = 16; # degree of polynomials 
tol = 1e-2 # tolerance for small eigenvalues
sol = optimize_square(d)
xp = LinRange(-1, 1, 500) 
yp = LinRange(-1, 1, 500) 
plt = plot_wrinkles(sol, xp, yp, tol, d)
plt[1] # show plot