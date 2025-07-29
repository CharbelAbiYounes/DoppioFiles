using LinearAlgebra

function basis(j,n)
    v = zeros(n)
    v[j] = 1.0
    v
end

function myLanczos(mat; k::Integer=size(mat,1), v::Vector{T1}=randn(size(mat,1)),opt::Integer=1) where T1<:Number
    m, n = size(mat)
    @assert m == n "Input matrix must be square"
    symTol = 1e-8
    @assert maximum(abs.(mat - transpose(mat))) ≤ symTol "Input matrix is not symmetric to working precision"
    Q = zeros(eltype(mat),n, k)
    q = v / norm(v)
    Q[:, 1] = q
    d = zeros(eltype(mat),k)
    od = zeros(eltype(mat),k-1)
    for i = 1:k
        z = mat * q
        d[i] = dot(q, z)
        if opt==1
            #Full re-orthogonalization (x2)
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
        elseif opt==2
            #Full re-orthogonalization
            z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
        else
            #No re-orthogonalization
            z -= d[i] * q
            if i > 1
                z -= od[i-1] * Q[:, i-1]
            end
        end
        if i < k
            od[i] = norm(z)
            if od[i]==0.0
                T = SymTridiagonal(d[1:i], od[1:i-1])
                return T,Q[:,1:i]
            end
            q = z / od[i]
            Q[:, i+1] = q
        end
    end
    T = SymTridiagonal(d, od)
    return T,Q
end

# A = Diagonal([5.0, 5.0, 5.0, 2.0])
# v = basis(1,4)

A = Diagonal([5.0, 5.0, 5.0, 5.0])
v = randn(4)

T,Q = myLanczos(A,v=v)
display(T)
display(eigvals(T))