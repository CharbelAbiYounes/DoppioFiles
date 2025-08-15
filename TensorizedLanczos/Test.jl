using LinearAlgebra
include("LanczosSpikeDetection.jl")

## Testing Lanczos with random vector on a matrix with repeated eigenvalues 

# function basis(j,n)
#     v = zeros(n)
#     v[j] = 1.0
#     v
# end

# function myLanczos(mat; k::Integer=size(mat,1), v::Vector{T1}=randn(size(mat,1)),opt::Integer=1) where T1<:Number
#     m, n = size(mat)
#     @assert m == n "Input matrix must be square"
#     symTol = 1e-8
#     @assert maximum(abs.(mat - transpose(mat))) ≤ symTol "Input matrix is not symmetric to working precision"
#     Q = zeros(eltype(mat),n, k)
#     q = v / norm(v)
#     Q[:, 1] = q
#     d = zeros(eltype(mat),k)
#     od = zeros(eltype(mat),k-1)
#     for i = 1:k
#         z = mat * q
#         d[i] = dot(q, z)
#         if opt==1
#             #Full re-orthogonalization (x2)
#             z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
#             z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
#         elseif opt==2
#             #Full re-orthogonalization
#             z -= Q[:, 1:i] * (Q[:, 1:i]' * z)
#         else
#             #No re-orthogonalization
#             z -= d[i] * q
#             if i > 1
#                 z -= od[i-1] * Q[:, i-1]
#             end
#         end
#         if i < k
#             od[i] = norm(z)
#             if od[i]==0.0
#                 T = SymTridiagonal(d[1:i], od[1:i-1])
#                 return T,Q[:,1:i]
#             end
#             q = z / od[i]
#             Q[:, i+1] = q
#         end
#     end
#     T = SymTridiagonal(d, od)
#     return T,Q
# end

# # A = Diagonal([5.0, 5.0, 5.0, 2.0])
# # v = basis(1,4)

# A = Diagonal([5.0, 5.0, 5.0, 5.0])
# v = randn(4)

# T,Q = myLanczos(A,v=v)
# display(T)
# display(eigvals(T))

## Block Lanczos with Incomplete Blocks

function block_lanczos(M, V, k; reorth::Bool=true)
    d, b = size(V)
    @assert d>b "Matrix already in desired form"
    ℓ = d%b
    max_iter = convert(Int64,ceil(d/b))
    floor_val = convert(Int64,floor(d/b))
    k = minimum([k,max_iter])
    if (k<floor_val) || (k>=floor_val&&ℓ==0)
        A = [zeros(b, b) for i in 1:k]
        B = [zeros(b, b) for i in 1:k]
        Q = zeros(d,b*(k+1))
    elseif k==floor_val
        A = [zeros(b, b) for i in 1:k]
        B = vcat([zeros(b, b) for i in 1:k-1],[zeros(ℓ,b)])
        Q = zeros(d,d)
    else
        A = vcat([zeros(b, b) for i in 1:k-1],[zeros(ℓ,ℓ)])
        B = vcat([zeros(b, b) for i in 1:k-2],[zeros(ℓ,b)],[zeros(ℓ,ℓ)])
        Q = zeros(d,d+ℓ)
    end 
    Q_tmp, B_0 = qr(V)
    sgn = Diagonal(sign.(diag(B_0)))
    B_0 = sgn*B_0
    Q_tmp = Q_tmp*sgn
    Q[:,1:b] = Matrix(Q_tmp)
    kk = minimum([k,floor_val-1])
    for j in 1:kk
        Qj = Q[:,(j-1)*b+1:j*b]
        if j==1
            Z = M*Qj
        else
            Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
            Z = M*Qj-Qjm1*(B[j-1])'
        end
        A[j] = Qj'*Z
        Z -= Qj*A[j]
        if reorth
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
        end
        Q_tmp, B[j] = qr(Z)
        sgn = Diagonal(sign.(diag(B[j])))
        B[j] = sgn*B[j]
        Q_tmp = Q_tmp*sgn
        Q[:, (j*b+1):(j+1)*b] = Matrix(Q_tmp)
    end
    if k>=floor_val&&ℓ==0
        j = k
        Qj = Q[:,(j-1)*b+1:j*b]
        if j==1
            Z = M*Qj
        else
            Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
            Z = M*Qj-Qjm1*(B[j-1])'
        end
        A[j] = Qj'*Z
        Z -= Qj*A[j]
        if reorth
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
        end
        Q_tmp, B[j] = qr(Z)
        sgn = Diagonal(sign.(diag(B[j])))
        B[j] = sgn*B[j]
        Q_tmp = Q_tmp*sgn
        Q[:, (j*b+1):(j+1)*b] = Matrix(Q_tmp)
    elseif k==floor_val
        j = k
        Qj = Q[:,(j-1)*b+1:j*b]
        if j==1
            Z = M*Qj
        else
            Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
            Z = M*Qj-Qjm1*(B[j-1])'
        end
        A[j] = Qj'*Z
        Z -= Qj*A[j]
        if reorth
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
        end
        Q_tmp, tmpB = qr(Z)
        Q_tmp = Matrix(Q_tmp)
        diag_B = diag(tmpB)
        sgn = Diagonal(sign.(diag_B))
        tmpB = sgn*tmpB
        Q_tmp = Q_tmp*sgn
        idx = abs.(diag_B).>=1e-10
        Q[:, (j*b+1):(j*b+ℓ)] = Q_tmp[:,idx]
        B[j] = tmpB[idx,:]
    elseif k>floor_val
        j = k-1
        Qj = Q[:,(j-1)*b+1:j*b]
        if j==1
            Z = M*Qj
        else
            Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
            Z = M*Qj-Qjm1*(B[j-1])'
        end
        A[j] = Qj'*Z
        Z -= Qj*A[j]
        if reorth
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
            Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
        end
        Q_tmp, tmpB = qr(Z)
        Q_tmp = Matrix(Q_tmp)
        diag_B = diag(tmpB)
        sgn = Diagonal(sign.(diag_B))
        tmpB = sgn*tmpB
        Q_tmp = Q_tmp*sgn
        idx = abs.(diag_B).>=1e-12
        Q[:, (j*b+1):(j*b+ℓ)] = Q_tmp[:,idx]
        B[j] = tmpB[idx,:]
        j = k
        Qj = Q[:,(j-1)*b+1:(j-1)*b+ℓ]
        Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
        Z = M*Qj-Qjm1*(B[j-1])'
        A[j] = Qj'*Z
        Z -= Qj*A[j]
        Q_tmp, B[j] = qr(Z)
        diag_B = diag(B[j])
        sgn = Diagonal(sign.(diag_B))
        B[j] = sgn*B[j]
        Q_tmp = Q_tmp*sgn
        Q[:, ((j-1)*b+ℓ+1):((j-1)*b+2*ℓ)] = Matrix(Q_tmp)
    end
    if (k<=floor_val)
        Qk = Q[:,1:b*k]
        Q_rem = Q[:,b*k+1:end]
        return Qk, Q_rem, A, B, B_0
    else
        Qk = Q[:,1:(k-1)*b+ℓ]
        Q_rem = Q[:,(k-1)*b+ℓ+1:end]
        return Qk, Q_rem, A, B, B_0
    end
end

function block_tridiag(A,B)
    r = size(A[1])[1]
    ℓ = 0
    k = length(A)
    flag = false
    if size(B[end])[2]!=r
        ℓ = size(B[end])[2]
        flag = true
    end
    if !flag
        T = zeros(Float64,k*r,k*r)
        for i=1:k
            T[r*(i-1)+1:r*i,r*(i-1)+1:r*i] = A[i]
        end
        for i=1:k-1
            T[r*i+1:r*(i+1),r*(i-1)+1:r*i] = B[i]
            T[r*(i-1)+1:r*i,r*i+1:r*(i+1)] = B[i]'
        end
    else
        T = zeros(Float64,(k-1)*r+ℓ,(k-1)*r+ℓ)
        for i=1:k-1
            T[r*(i-1)+1:r*i,r*(i-1)+1:r*i] = A[i]
        end
        T[r*(k-1)+1:end,r*(k-1)+1:end] = A[end]
        for i=1:k-2
            T[r*i+1:r*(i+1),r*(i-1)+1:r*i] = B[i]
            T[r*(i-1)+1:r*i,r*i+1:r*(i+1)] = B[i]'
        end
        T[r*(k-1)+1:r*(k-1)+ℓ,r*(k-2)+1:r*(k-1)] = B[k-1]
        T[r*(k-2)+1:r*(k-1),r*(k-1)+1:r*(k-1)+ℓ] = B[k-1]'
    end
    return T
end

N = 9
k = 2
diag0 = randn(N)
diag1 = randn(N-1)
diag2 = abs.(randn(N-2))
W = diagm(0=>diag0,1=>diag1,2=>diag2,-1=>diag1,-2=>diag2)
W = Symmetric(W)
V = zeros(Float64,N,k)
for i=1:k
    V[i,i]=1.0
end
Qk, Q_rem, A, B, B_0 = block_lanczos(W, V, 5)
T = block_tridiag(A,B)

display(W)
display(T)