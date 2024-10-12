using JuMP
using SumOfSquares
using CSDP
using DynamicPolynomials
using QuadGK
using FastGaussQuadrature, LinearAlgebra


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
    model = SOSModel(CSDP.Optimizer)
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
    model = SOSModel(CSDP.Optimizer)
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
## RUN SOME TESTS
################################################################################
d = 6;
sol = optimize_square(d)
#sol = optimize_ellipse(d)




# Define the range for x and y using LinRange
xp = LinRange(-1, 1, 31) 
yp = LinRange(-1, 1, 31) 



# Extract individual polynomials from the matrix
a = sol[4][1, 1]
bb = sol[4][1, 2]
c = sol[4][2, 1]
d = sol[4][2, 2]
# μ = sol[4];
xy = sol[1];

######################## METHOD 1: Separate loops ####################################

resultList = []
combined_result = []
eigen_value_list = []
eigen_vector_list = []
for x in xp
    for y in yp

        results_a = a(x, y)
        results_bb = bb(x, y)
        results_c = c(x, y)
        results_d = d(x, y)

        combined_result=[results_a results_bb; results_c results_d]
        push!(resultList, combined_result)

    end
end 


# Compute eigen values and vectors
eigen_value_list = []
eigen_vector_list = []
for index_lm in 1:length(resultList)
    # Calculate eigen matrix of single evaluated matrix take from resultList
    matrix = resultList[index_lm]
    result = eigen(matrix)

    # We store each eigenvalue (2 x 1vector) and corresponding eigenvector (2 x 2 matrix)
    push!(eigen_value_list, result.values)
    push!(eigen_vector_list, result.vectors)
end


clean_eigen_values_list = []
clean_eigen_vector_list = []

for index_pk in 1:length(eigen_value_list)
    
    if eigen_value_list[index_pk][1] < 1e-2
       push!(clean_eigen_vector_list, eigen_vector_list[index_pk][:,1])

    elseif eigen_value_list[index_pk][2] < 1e-2
        push!(clean_eigen_vector_list, eigen_vector_list[index_pk][:,2])
    end
    
end

u = Float64[]
v = Float64[]
for index_ak in 1:length(clean_eigen_vector_list)
    push!(u,clean_eigen_vector_list[index_ak][2])
    push!(v,-clean_eigen_vector_list[index_ak][1])
end

############################################################
############################################################



################# METHOD 2 = 1 loop #################

resultList = []
combined_result = []

eigen_value_list = []
eigen_vector_list = []

u = zeros(length(xp), length(yp))
v = zeros(length(xp), length(yp))

eigen_matrix = []

i = 1

for x in xp
    j = 1
    for y in yp

        results_a = a(x, y)
        results_bb = bb(x, y)
        results_c = c(x, y)
        results_d = d(x, y)

        combined_result=[results_a results_bb; results_c results_d]
        push!(resultList, combined_result)

        eigen_matrix = eigen(combined_result)
        push!(eigen_value_list, eigen_matrix.values)
        push!(eigen_vector_list, eigen_matrix.vectors)

        if eigen_matrix.values[1] < 1e-2
            # push!(threshold_eigen_vector_list, eigen_matrix.vectors[:,1])
            v[i, j] = -eigen_matrix.vectors[1, 1]
            u[i, j] = eigen_matrix.vectors[2, 1]
        end

        if eigen_matrix.values[2] < 1e-2
            # push!(threshold_eigen_vector_list, eigen_matrix.vectors[:,2])
            v[i, j] = -eigen_matrix.vectors[1, 2]
            u[i, j] = eigen_matrix.vectors[2, 2]
        end

        counter+=1
        j+=1
    end
    i+=1
end 



# for i in 1:size(u, 1)
#     for j in 1:size(u, 2)
#         if u[i, j] == 0.0
#             u[i,j]== NaN
#         end

#        if v[i, j] == 0.0
#             v[i,j]== NaN
#         end

#     end
# end


############ Plotting ######################

############### 1st way ####################
using CairoMakie
# Create a vector field plot
fig, ax, qvr = quiver(xp, yp, u, v, colormap=colormap, colorrange=(0.0, 1.0), linewidth=2)

# Customize the plot
ax.title = "Vector Field Plot"
ax.xlabel = "x-axis"
ax.ylabel = "y-axis"

# Display the plot
display(fig)

################# 2nd way ##################
using CairoMakie
f = Figure(size = (400, 400))
Axis(f[1, 1], backgroundcolor = "black")
strength = vec(sqrt.(u .^ 2 .+ v .^ 2))
arrows!(xp, yp, u, v, arrowsize = 10, lengthscale = 0.3, arrowcolor = strength)
f




# u = Float64[]
# v = Float64[]
# for index_ak in 1:length(threshold_eigen_vector_list)
#     push!(u,threshold_eigen_vector_list[index_ak][2])
#     push!(v,-threshold_eigen_vector_list[index_ak][1])
# end


# using Plots
# # Plots.quiver(xp,yp,quiver=(u,v))


# # Plot the vector field using quiver
# Plots.quiver(xp, yp, quiver=(u, v), title="Vector Field", xlabel="x", ylabel="y", aspect_ratio=:equal)

# # Display the plot
# Plots.plot!()












