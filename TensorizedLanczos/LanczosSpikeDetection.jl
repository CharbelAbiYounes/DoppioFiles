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
            if od[i]==0
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

function Cholesky(T::SymTridiagonal)
    n = size(T)[1]
    L = Tridiagonal(copy(T))
    for k = 1:n-1
        L[k+1,k+1] = L[k+1,k+1] - L[k+1,k]^2/L[k,k]
        L[k:k+1,k] = L[k:k+1,k]/sqrt(L[k,k])
        L[k,k+1] = 0.
    end
    L[n,n] = sqrt(L[n,n])
    return L
end

function ConvChol(mat,tol::Float64,seq_len::Integer,jmp::Integer,max_iter::Integer;v=randn(size(mat,1)),orth::Bool=true)
    m, n = size(mat)
    Q = zeros(eltype(mat),n, max_iter)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),max_iter)
    od = zeros(eltype(mat),max_iter-1)
    z = similar(q)
    dAvrg_old, odAvrg_old, dStd_old, odStd_old = zeros(Float64,4)
    idx = seq_len
    i=1
    Convflag = false
    while i<=max_iter && !Convflag 
        z .= mat * q
        d[i] = dot(q, z)
        if orth
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < max_iter
            od[i] = norm(z)
            q .= z / od[i]
            Q[:, i+1] .= q
            if i==idx 
                dAvrg = sum(@view d[i-seq_len+1:i])/seq_len
                odAvrg = sum(@view od[i-seq_len+1:i])/seq_len
                dStd = sqrt(sum(((@view d[i-seq_len+1:i]).-dAvrg).^2)/(seq_len-1))
                odStd = sqrt(sum(((@view od[i-seq_len+1:i]).-odAvrg).^2)/(seq_len-1))
                if dStd<tol && odStd<tol && dStd_old<tol && odStd_old<tol && abs(dAvrg-dAvrg_old)<tol && abs(odAvrg-odAvrg_old)<tol
                    z .= mat * q
                    d[i+1] = dot(q, z)
                    Convflag = true
                else
                    dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
                    idx = idx+jmp
                end
            end
        end
        i+=1
    end
    i = min(i,max_iter)
    return Convflag, Cholesky(SymTridiagonal(d[1:i],od[1:i-1]))
end

function CholSeq(mat,tol::Float64,seq_len::Integer,jmp::Integer,max_iter::Integer,vecNbr::Integer;Modtol::Float64=tol,vecList=randn(size(mat,1),vecNbr))
    ModChol = Vector{Matrix{Float64}}(undef, vecNbr)
    TrueChol = Vector{Matrix{Float64}}(undef, vecNbr)
    sizelist = zeros(Int64,vecNbr)
    d = zeros(Float64,max_iter)
    od = zeros(Float64,max_iter)
    for j=1:vecNbr
        Convflag, TrueChol[j] = ConvChol(mat,tol,seq_len,jmp,max_iter,v=vecList[:,j])
        L = TrueChol[j]
        idx = size(L,1)-1
        d[1:idx+1] .= diag(L,0)
        od[1:idx] .= diag(L,-1)
        i,d_Sum,od_Sum,dAsymp,odAsymp = zeros(Float64,5)
        if Convflag
            i = idx-jmp-seq_len
            d_Sum= sum(@view d[idx-jmp-seq_len+1:idx+1])
            od_Sum= sum(@view od[idx-jmp-seq_len+1:idx])
            dAsymp = d_Sum/(jmp+seq_len+1)
            odAsymp = od_Sum/(jmp+seq_len)
            while i>0 && abs(d[i]-dAsymp)<Modtol && abs(od[i]-odAsymp)<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(idx-i+1)
            odAsymp = od_Sum/(idx-i)
        else
            i = max_iter-2
            d_Sum = d[max_iter-1]
            od_Sum = od[max_iter-1]
            while abs(d[i]-d[max_iter-1])<Modtol && abs(od[i]-od[max_iter-1])<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(max_iter-1-i)
            odAsymp = od_Sum/(max_iter-1-i)
        end
        d[i+1], d[i+2], od[i+1] = dAsymp, dAsymp, odAsymp
        ModChol[j] = Tridiagonal(od[1:i+1],d[1:i+2],0*od[1:i+1])
        sizelist[j] = i+2
    end
    dAsymp = 0
    odAsymp = 0
    for j=1:vecNbr
        dAsymp += (ModChol[j][sizelist[j],sizelist[j]]) + (ModChol[j][sizelist[j]-1,sizelist[j]-1])
        odAsymp += (ModChol[j][sizelist[j],sizelist[j]-1])
    end
    dAsymp/=(2*vecNbr)
    odAsymp/=vecNbr
    for j=1:vecNbr
        ModChol[j][sizelist[j],sizelist[j]] = dAsymp
        ModChol[j][sizelist[j]-1,sizelist[j]-1] = dAsymp
        ModChol[j][sizelist[j],sizelist[j]-1] = odAsymp
    end
    return TrueChol,ModChol
end

BidiagRel = (m,z,d,ℓ) -> 1/(-z+d^2-d^2*ℓ^2*(m/(1+ℓ^2*m)))
BidiagRef = (z,d,ℓ) -> (-z+d^2-ℓ^2+sqrt(z-(d+ℓ)^2)*sqrt(z-(d-ℓ)^2))/(2*z*ℓ^2)
function BidiagRec(z::Number,L,N::Integer;eps::Float64=1e-3)
    n = size(L,1)
    m0 = BidiagRef(z+1im*eps,L[n-1,n-1],L[n,n-1])
    for j = n-2:-1:1
            m0 = BidiagRel(m0,z+1im*eps,L[j,j],L[j+1,j])
    end
    return m0
end

function EstimSupp(ModChol)
    n = size(ModChol[1],1)
    dAsymp = ModChol[1][n,n]
    odAsymp = ModChol[1][n,n-1]
    γmin = (dAsymp-odAsymp)^2
    γplus = (dAsymp+odAsymp)^2
    return γmin, γplus
end

function EstimDensity(x,ModChol,N::Integer)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    len = length(ModChol)
    for i=1:len_x
        for j=1:len
            dens[i]+=imag(BidiagRec(x[i],ModChol[j],N))/π
        end
    end
    return dens/len
end

function EstimSpike(TrueChol,N::Integer;δ::Float64=0.25,c::Float64=1.0,ThreshOut::Bool=false)
    len = length(TrueChol)
    Vec = zeros(Float64,len)
    sizelist = zeros(Int64,len)
    γplus = (TrueChol[1][end,end]+TrueChol[1][end,end-1])^2
    for i=1:len
        L = TrueChol[i]
        evals = eigvals(SymTridiagonal(L*L'))
        j = size(L,1)
        sizelist[i] = j
        while j>1 && evals[j]>γplus+c*N^(-δ)
            Vec[i] += 1
            j-=1
        end
    end
    freq = Dict{Int, Int}()
    for v in Vec
        freq[v] = get(freq, v, 0) + 1
    end
    Nbr = findmax(freq)[2]
    maxval,maxidx = findmin(sizelist)
    L = TrueChol[maxidx]
    evals = eigvals(SymTridiagonal(L*L'))
    if !ThreshOut
        return Nbr, evals[end-Nbr+1:end]
    else
        return Nbr, evals[end-Nbr+1:end], γplus+c*N^(-δ)
    end
end

function KronMult(A, x)
    N = size(A, 1)
    K = length(x) ÷ N
    y = similar(x)
    for i in 0:K-1
        y[(i*N+1):(i+1)*N] .= A * view(x, (i*N+1):(i+1)*N)
    end
    return y
end

function KronConvChol(mat,tol::Float64,seq_len::Integer,jmp::Integer,max_iter::Integer,v;orth::Bool=true)
    N = size(mat, 1)
    @assert length(v) % N==0 "Dimensions don't match!"
    K = length(v) ÷ N
    Q = zeros(eltype(mat),K*N, max_iter)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),max_iter)
    od = zeros(eltype(mat),max_iter-1)
    z = similar(q)
    dAvrg_old, odAvrg_old, dStd_old, odStd_old = zeros(Float64,4)
    idx = seq_len
    i=1
    Convflag = false
    while i<=max_iter && !Convflag 
        z .= KronMult(mat,q)
        d[i] = dot(q, z)
        if orth
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < max_iter
            od[i] = norm(z)
            q .= z / od[i]
            Q[:, i+1] .= q
            if i==idx 
                dAvrg = sum(@view d[i-seq_len+1:i])/seq_len
                odAvrg = sum(@view od[i-seq_len+1:i])/seq_len
                dStd = sqrt(sum(((@view d[i-seq_len+1:i]).-dAvrg).^2)/(seq_len-1))
                odStd = sqrt(sum(((@view od[i-seq_len+1:i]).-odAvrg).^2)/(seq_len-1))
                if dStd<tol && odStd<tol && dStd_old<tol && odStd_old<tol && abs(dAvrg-dAvrg_old)<tol && abs(odAvrg-odAvrg_old)<tol
                    z .= KronMult(mat,q)
                    d[i+1] = dot(q, z)
                    Convflag = true
                else
                    dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
                    idx = idx+jmp
                end
            end
        end
        i+=1
    end
    i = min(i,max_iter)
    return Cholesky(SymTridiagonal(d[1:i],od[1:i-1]))
end

function KronEstimSupp(KronChol)
    n = size(KronChol, 1)
    γmin = (KronChol[n,n]-KronChol[n,n-1])^2
    γplus = (KronChol[n,n]+KronChol[n,n-1])^2
    return γmin, γplus   
end

function KronEstimDensity(x,KronChol,N::Integer)
    len = length(x)
    dens = zeros(Float64,len)
    for i=1:len 
        dens[i]=imag(BidiagRec(x[i],KronChol,N))/π
    end
    return dens
end

function KronEstimSpike(KronChol,γplus,N;δ::Float64=0.25,c::Float64=1.0)
    Nbr = 0
    evals = eigvals(SymTridiagonal(KronChol*KronChol'))
    i = size(KronChol, 1)
    while i > 1 && evals[i] > γplus + c * N^(-δ)
        Nbr += 1
        i -= 1
    end
    return Nbr, evals[end - Nbr + 1:end]
end

function block_tridiag(A,B)
    r = size(A[1],1)
    ℓ = 0
    k = length(A)
    flag = false
    if size(B[end],1)!=r
        ℓ = size(B[end],1)
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

function block_bidiag(C,D)
    r = size(C[1])[1]
    ℓ = 0
    k = length(C)
    flag = false
    if size(D[end],1)!=r
        ℓ = size(D[end],1)
        flag = true
    end
    if !flag
        L = zeros(Float64,k*r,k*r)
        for i=1:k
            L[r*(i-1)+1:r*i,r*(i-1)+1:r*i] = C[i]
        end
        for i=1:k-1
            L[r*i+1:r*(i+1),r*(i-1)+1:r*i] = D[i]
        end
    else
        L = zeros(Float64,(k-1)*r+ℓ,(k-1)*r+ℓ)
        for i=1:k-1
            L[r*(i-1)+1:r*i,r*(i-1)+1:r*i] = C[i]
        end
        L[r*(k-1)+1:end,r*(k-1)+1:end] = C[end]
        for i=1:k-2
            L[r*i+1:r*(i+1),r*(i-1)+1:r*i] = D[i]
        end
        L[r*(k-1)+1:r*(k-1)+ℓ,r*(k-2)+1:r*(k-1)] = D[end]
    end
    return L
end

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

function FindSteadyState(A,B,init_tol,max_idx,seq_len)
    tol = init_tol
    flag_tol = true
    while(flag_tol)
        n = 0
        diagn = A[n+2:n+seq_len+1].-A[n+1:n+seq_len]
        offdiagn = B[n+2:n+seq_len].-B[n+1:n+seq_len-1]
        diagn = [opnorm(diagn[i]) for i=1:seq_len]
        offdiagn = [opnorm(offdiagn[i]) for i=1:seq_len-1]
        diagn_std = std(diagn)
        off_std = std(offdiagn)
        flag = diagn_std>tol || off_std>tol
        while (flag && n+seq_len+2≤ max_idx)
            n = n+1
            diagn = A[n+2:n+seq_len+1].-A[n+1:n+seq_len]
            offdiagn = B[n+2:n+seq_len].-B[n+1:n+seq_len-1]
            diagn = [opnorm(diagn[i]) for i=1:seq_len]
            offdiagn = [opnorm(offdiagn[i]) for i=1:seq_len-1]
            diagn_std = std(diagn)
            off_std = std(offdiagn)
            if diagn_std<tol && off_std<tol 
                flag = false
            end
        end
        if !(flag)
            return n
        else
            tol = tol+init_tol    
        end
    end
end

function ModChol1(C,D)
    b = size(C[1],1)
    n = length(C)
    ModC = copy(C)
    ModD = copy(D)
    L = block_bidiag(C[n-1:n],[D[n-1]])
    T = L*L'
    T = Symmetric(T)
    tridiag, _ = myLanczos(T,v=basis(1,2*b))
    bidiag = Cholesky(tridiag)
    d = bidiag[1,1]
    od = bidiag[2,1]
    ModC[n-1] = Diagonal(d*ones(Float64,b))
    ModC[n] = Diagonal(d*ones(Float64,b))
    ModD[n-1] = Diagonal(od*ones(Float64,b))
    return ModC, ModD
end

function ModChol2(C,D)
    b = size(C[1],1)
    n = length(C)
    ModC = copy(C)
    ModD = copy(D)
    d = sum(diag(C[n]+C[n-1]))/(2*b)
    od = sum(diag(D[n-1]))/b
    ModC[n-1] = Diagonal(d*ones(Float64,b))
    ModC[n] = Diagonal(d*ones(Float64,b))
    ModD[n-1] = Diagonal(od*ones(Float64,b))
    return ModC, ModD
end

BlockBidiagRel = (M,z,C,D) -> inv(C*C' - z*I - C*D'*(M - M*inv(inv(D*D') + M)*M)*D*C')

function BlockBidiagRec1(z::Number,ModC,ModD,N::Integer;eps::Float64=1e-3)
    b = size(ModC[1],1)
    n = length(ModC)
    m0 = BidiagRef(z+1im*eps,ModC[n-1][1,1],ModD[n-1][1,1])
    M0 = Diagonal(m0*ones(Float64,b))
    for j = n-2:-1:1
        M0 = BlockBidiagRel(M0,z+1im*eps,ModC[j],ModD[j])
    end
    return M0
end

function BlockBidiagRec2(z::Number,C,D,N::Integer;eps::Float64=1e-3,imax=500,tol::Float64=1e-8,f::Function=x->1)
    b = size(C[1],1)
    n = length(C)
    M0 = 0.1*ones(b,b)
    Mold = M0
    for i=1:imax
        α = min(f(n),1)
        M0 = α*BlockBidiagRel(M0,z+1im*eps,C[n-1],D[n-1]) + (1-α)*M0
        max_term = maximum(abs.(M0-Mold))
        if max_term<tol
            flag = false
            break
        end
        Mold = M0
    end
    for j = n-2:-1:1
        M0 = BlockBidiagRel(M0,z+1im*eps,C[j],D[j])
    end
    return M0
end

function BlockEstimSupp(ModC,ModD)
    dAsymp = ModC[end][1,1]
    odAsymp = ModD[end][1,1]
    γmin = (dAsymp-odAsymp)^2
    γplus = (dAsymp+odAsymp)^2
    return γmin, γplus
end

function BlockEstimDensity1(x,C,D,N::Integer)
    b = size(C[1],1)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    for i=1:len_x
        dens[i]+=tr(imag(BlockBidiagRec1(x[i],C,D,N))/π)/b
    end
    return dens
end

function BlockEstimDensity2(x,C,D,N::Integer)
    b = size(C[1],1)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    for i=1:len_x
        dens[i]+=tr(imag(BlockBidiagRec2(x[i],C,D,N))/π)/b
    end
    return dens
end

function BlockEstimSpike(C,D,γplus,N::Integer;δ::Float64=0.25,c::Float64=1.0,ThreshOut::Bool=false)
    L = block_bidiag(C,D)
    T = Symmetric(L*L')
    evals = eigvals(T)
    n = size(L,1)
    Nbr = 0
    while n>1 && evals[n]>γplus+c*N^(-δ)
        Nbr+=1
        n-=1
    end
    return Nbr, evals[end-Nbr+1:end]
end

# function ConvBlockChol(mat, tol::Float64, seq_len::Integer, jmp::Integer, max_iter, V; reorth::Bool=true)
#     d, b = size(V)
#     @assert d>b "Matrix already in desired form"
#     A = []
#     B = []
#     Q, B_0 = qr(V)
#     sgn = Diagonal(sign.(diag(B_0)))
#     B_0 = sgn*B_0
#     Q = Q*sgn
#     Q = Matrix(Q)
#     dAvrg_old = zeros(Float64,b,b)
#     odAvrg_old = zeros(Float64,b,b)
#     dStd_old = zeros(Float64,b,b)
#     odStd_old =zeros(Float64,b,b)
#     idx = seq_len
#     j=1
#     Convflag = false
#     while j<=max_iter && !Convflag 
#         Qj = Q[:,(j-1)*b+1:j*b]
#         if j==1
#             Z = mat*Qj
#         else
#             Qjm1 = Q[:,(j-2)*b+1:(j-1)*b]
#             Z = mat*Qj-Qjm1*(B[j-1])'
#         end
#         push!(A,Qj'*Z)
#         Z -= Qj*A[j]
#         if reorth
#             Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
#             Z -= Q[:,1:(j-1)*b]*(Q[:,1:(j-1)*b]'*Z)
#         end
#         if j < max_iter
#             Q_tmp, B_tmp = qr(Z)
#             sgn = Diagonal(sign.(diag(B_tmp)))
#             B_tmp = sgn*B_tmp
#             Q_tmp = Q_tmp*sgn
#             push!(B,B_tmp)
#             Q = hcat(Q,Matrix(Q_tmp))
#             # if j>seq_len
#             #     diagn = A[j-seq_len+1:j].-A[j-seq_len:j-1]
#             #     offdiagn = B[j-seq_len+1:j].-B[j-seq_len:j-1]
#             #     diagn = [opnorm(diagn[i]) for i=1:length(diagn)]
#             #     offdiagn = [opnorm(offdiagn[i]) for i=1:length(offdiagn)]
#             #     diagn_std = std(diagn)
#             #     off_std = std(offdiagn)
#             #     if (diagn_std<tol && off_std<tol) 
#             #         Convflag = true
#             #         Qj = Q[:,j*b+1:(j+1)*b]
#             #         Qjm1 = Q[:,(j-1)*b+1:j*b]
#             #         Z = mat*Qj-Qjm1*(B[j])'
#             #         push!(A,Qj'*Z)
#             #     end
#             # end
#             if j==idx
#                 tmp_sum = 0*dAvrg_old
#                 for i=j-seq_len+1:j
#                     tmp_sum += A[i]/seq_len
#                 end
#                 dAvrg = tmp_sum
#                 tmp_sum = 0*odAvrg_old
#                 for i=j-seq_len+1:j
#                     tmp_sum += B[i]/seq_len
#                 end
#                 odAvrg = tmp_sum
#                 tmp_sum = 0*dStd_old
#                 for i=j-seq_len+1:j
#                     tmp_sum += (A[i]-dAvrg).^2 /(seq_len-1) 
#                 end
#                 dStd = sqrt.(tmp_sum)
#                 tmp_sum = 0*odStd_old
#                 for i=j-seq_len+1:j
#                     tmp_sum += (B[i]-dAvrg).^2/(seq_len-1) 
#                 end
#                 odStd = sqrt.(tmp_sum)
#                 if norm(dStd)<tol && norm(odStd)<tol && norm(dStd_old)<tol && norm(odStd_old)<tol && norm(dAvrg-dAvrg_old)<tol && norm(odAvrg-odAvrg_old)<tol
#                     Qj = Q[:,j*b+1:(j+1)*b]
#                     Qjm1 = Q[:,(j-1)*b+1:j*b]
#                     Z = mat*Qj-Qjm1*(B[j])'
#                     push!(A,Qj'*Z)
#                     Convflag = true
#                 else
#                     dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
#                     idx = idx+jmp
#                 end
#             end
#         end
#         j+=1
#     end
#     j = min(j,max_iter)
#     T = block_tridiag(A[1:j],B[1:j-1])
#     T = (T+T')/2
#     Chol = cholesky(T)
#     L = Chol.L
#     C = [zeros(Float64,b,b) for _ in 1:j]
#     D = [zeros(Float64,b,b) for _ in 1:j-1]
#     for i=1:j-1
#         C[i] = L[(i-1)*b+1:i*b,(i-1)*b+1:i*b]
#         D[i] = L[i*b+1:(i+1)*b,(i-1)*b+1:i*b]
#     end
#     C[j] = L[(j-1)*b+1:j*b,(j-1)*b+1:j*b]
#     # d = sum(diag(C[j]+C[j-1]))/(2*b)
#     # od = sum(diag(D[j-1]))/b
#     L = block_bidiag(C[j-1:j],[D[j-1]])
#     T = L*L'
#     T = Symmetric(T)
#     tridiag, _ = myLanczos(T,v=basis(1,2*b))
#     bidiag = Cholesky(tridiag)
#     d = bidiag[1,1]
#     od = bidiag[2,1]
#     C[j-1] = Diagonal(d*ones(Float64,b))
#     C[j] = Diagonal(d*ones(Float64,b))
#     D[j-1] = Diagonal(od*ones(Float64,b))
#     return C, D
# end