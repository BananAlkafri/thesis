
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
import CairoMakie as cm 





    # Extend the definition of u to include more elements
    Dual_model = SOSModel(Mosek.Optimizer)
    # Define the problem
    @polyvar x y
    MON = monomials([x, y], 0:d)
    @variable(Dual_model,σ₁₁, Poly(MON))
    @variable(Dual_model,σ₁₂, Poly(MON))
    @variable(Dual_model,σ₂₂, Poly(MON))
    σ=[σ₁₁ σ₁₂; σ₁₂ σ₂₂]
    p = x^2 + y^2
    grad_P= differentiate(p, (x, y))
    tensor_product= (∇p * ∇p')
    Identity= Matrix{Float64}(I, 2, 2)
    dot(tensor_product, Identity-σ)
    # int_gradp = integrate_polynomial_square(∇p_squared)
    @constraint(model, differentiate(σ₁₁, x) + differentiate(σ₁₂, y)=0, domain = K)
    @constraint(model, differentiate(σ₁₂, x) + differentiate(σ₂₂, y)=0, domain = K)
    # ∇p = (∇p * ∇p')
    K = @set 1 - x^2 ≥ 0 && 1 - y^2 ≥ 0
    @constraint(model, μ in PSDCone(), domain = K)
    @objective(model, Max, )
    optimize!(model)
 