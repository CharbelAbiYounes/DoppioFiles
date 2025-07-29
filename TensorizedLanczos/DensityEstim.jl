using LinearAlgebra, Plots, LaTeXStrings
include("LanczosSpikeDetection.jl")
include("AuxiliaryFunctions.jl")

N = 5000
d = 0.1
M = convert(Int64,ceil(N/d))
X = randn(N,M)
K = 200
nodes,weights = Legendre(K)
a = 0.1
b = 4
h = x-> a<x<b ? (x^4+1)*sqrt(x-a)*sqrt(b-x)/x^2 : 0
normCst = QuadInt(h,a,b,nodes,weights)
scaled_h = x->h(x)/normCst
quantiles = zeros(Float64,N+1)
quantiles[1] = a
quantiles[N+1] = b
for i=2:N
    QuantEq = x->QuadInt(scaled_h,a,x,nodes,weights)-(i-1)/N
    quantiles[i] = Bisection(QuantEq,quantiles[1],quantiles[N+1])
end
sqrtΣ = Diagonal(sqrt.(quantiles[1:end-1]))
W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
evals = eigvals(W)

vecNbr = 200
vecList = randn(N,vecNbr)

# No Averaging
k = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 2.1/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr,vecList=vecList)
γmin, γplus = EstimSupp([TChol[1]])
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,[TChol[1]],N)
p = histogram(evals,bins=γmin-0.2:0.1:γplus+0.2,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="No Average")
savefig(p, "DensNoAvrg.pdf")

display("No Average "*string(size(TChol[1], 1)))

# Averaging
γmin, γplus = EstimSupp(L_list)
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,L_list,N)
p = histogram(evals,bins=γmin-0.2:0.1:γplus+0.2,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Average")
savefig(p, "DensAvrg.pdf")

sizelist = zeros(Float64,length(L_list))
for i=1:length(sizelist)
    sizelist[i] = size(L_list[i], 1)
end
display("Average "*string(maximum(sizelist)))

# Tensorized Average
function KronMult(A, x)
    N = size(A, 1)
    K = length(x) ÷ N
    y = similar(x)
    for i in 0:K-1
        y[(i*N+1):(i+1)*N] .= A * view(x, (i*N+1):(i+1)*N)
    end
    return y
end
function CholeskyKron(mat,tol::Float64,k::Integer,jmp::Integer,max_iter::Integer,v;orth::Bool=true)
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
    idx = k
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
            if od[i]==0
                return Cholesky(SymTridiagonal(d[1:i],od[1:i-1]))
            end
            q .= z / od[i]
            Q[:, i+1] .= q
            if i==idx 
                dAvrg = sum(@view d[i-k+1:i])/k
                odAvrg = sum(@view od[i-k+1:i])/k
                dStd = sqrt(sum(((@view d[i-k+1:i]).-dAvrg).^2)/(k-1))
                odStd = sqrt(sum(((@view od[i-k+1:i]).-odAvrg).^2)/(k-1))
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

tallvec = vec(vecList)
CholKron = CholeskyKron(W,tol,k,jmp,max_iter,tallvec)
n = size(CholKron, 1)
γmin = (CholKron[n,n]-CholKron[n,n-1])^2
γplus = (CholKron[n,n]+CholKron[n,n-1])^2
x = -0.1+γmin:0.001:γplus+0.1
len_x = length(x)
densKron = zeros(Float64,len_x)
for i=1:len_x
    densKron[i]=imag(BiRec(x[i],CholKron,N))/π
end
p = histogram(evals,bins=γmin-0.2:0.1:γplus+0.2,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,densKron,linecolor=:red,linewidth=3,label="Average")
savefig(p, "DensKron.pdf")

display("Kron Average "*string(n))