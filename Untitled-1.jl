

A = [1 2; 3 4];
B = [5 6; 7 8];
k= kron(A, B)
c=dot(A, B)

f=Matrix(1I, 2, 2)
f1=I(2)

Matrix(1I, 3, 3)    # Identity matrix of Int type
Matrix(1.0I, 3, 3)  # Identity matrix of Float64 type
Matrix(I, 3, 3)     # Identity matrix of Bool type

Matrix{Float64}(I, 2, 2)

ceil(Int, 1/4)