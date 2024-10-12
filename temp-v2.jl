resultList = []
combined_result = []

eigen_value_list = []
eigen_vector_list = []

threshold_eigen_vector_list = []
threshold_eigen_vector_list = []


for x in xp
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
            push!(threshold_eigen_vector_list, eigen_matrix.vectors[:,1])
            # push!(v, -eigen_matrix.vectors[:,1])
     
        elseif eigen_matrix.values[2] < 1e-2
             push!(threshold_eigen_vector_list, eigen_matrix.vectors[:,2])
            #  push!(u, eigen_matrix.vectors[:,2])
         end

    end
end 

u = Float64[]
v = Float64[]
for index_ak in 1:length(threshold_eigen_vector_list)
    push!(u,threshold_eigen_vector_list[index_ak][2])
    push!(v,-threshold_eigen_vector_list[index_ak][1])
end