using LinearAlgebra, Plots, LaTeXStrings
include("LanczosSpikeDetection.jl")
include("AuxiliaryFunctions.jl")

N = 3000
d = 0.1
M = convert(Int64,ceil(N/d))
X = randn(N,M)
K = 200
nodes,weights = Legendre(K)
a = 0.1
b = 4
# h = x-> a<x<b ? (x^4+1)*sqrt(x-a)*sqrt(b-x)/x^2 : 0
h = x-> a<x<b ? (2*(3.5-x)^3+x)*(b-x)^(1/2)*(x-a)^(1/2)/(2*(4.5-x)^2) : 0
normCst = QuadInt(h,a,b,nodes,weights)
scaled_h = x->h(x)/normCst
quantiles = zeros(Float64,N+1)
quantiles[1] = a
quantiles[N+1] = b
for i=2:N
    QuantEq = x->QuadInt(scaled_h,a,x,nodes,weights)-(i-1)/N
    quantiles[i] = Bisection(QuantEq,quantiles[1],quantiles[N+1])
end
spikes = [6,6.4,6.6,6.9,7,7.5,7.7,8]
quantiles[1:length(spikes)] = spikes
sqrtΣ = Diagonal(sqrt.(quantiles[1:end-1]))
W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric

vecNbr = 100
vecList = randn(N,vecNbr)

# No Averaging
seq_len = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 2.1/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
TrueChol, ModChol = CholSeq(W,tol,seq_len,jmp,max_iter,vecNbr;vecList=vecList)
p1 = plot(diag(TrueChol[1],0),linecolor=:salmon,linewidth=1,alpha=0.1,label="No Average Diag",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
for i=1:length(ModChol)
    p1 = plot!(diag(TrueChol[i],0),linecolor=:salmon,linewidth=1,alpha=0.1,label="")
end
p1 = plot!(diag(TrueChol[1],-1),linecolor=:lightblue,linewidth=1,alpha=0.1,label="No Average Off Diag")
for i=1:length(ModChol)
    p1 = plot!(diag(TrueChol[i],-1),linecolor=:lightblue,linewidth=1,alpha=0.1,label="")
end

# Tensorized Average
tallvec = vec(vecList)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
n1 = size(KronChol, 1)
p1 = plot!(diag(KronChol,0),linecolor=:darkred,linewidth=3,label="Kron Diag")
p1 = plot!(diag(KronChol,-1),linecolor=:darkblue,linewidth=3,label="Kron Off Diag")

# Alternative Tensorized Average
Vec = copy(vecList)
for i=1:size(Vec,2)
    Vec[:,i] = Vec[:,i]/norm(Vec[:,i])
end
tallvec = vec(Vec)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
n2 = size(KronChol, 1)
p1 = plot!(diag(KronChol,0),linecolor=:darkorange,linestyle=:dash,linewidth=2,label="Alt Kron Diag")
p1 = plot!(diag(KronChol,-1),linecolor=:darkgreen,linestyle=:dash,linewidth=2,label="Alt Kron Off Diag")

# Trace Measure (Kronecker)
Q, R = qr(vecList)
sgn = Diagonal(sign.(diag(R)))
R = sgn*R
Q = Q*sgn
Q = Matrix(Q)
tallvec = vec(Q)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
n3 = size(KronChol, 1)
p1 = plot!(diag(KronChol,0),linecolor=:firebrick,linestyle=:dashdotdot,linewidth=2,label="Trace/Kron Diag")
p1 = plot!(diag(KronChol,-1),linecolor=:midnightblue,linestyle=:dashdotdot,linewidth=2,label="Trace/Kron Off Diag")

xlims!(p1, 0, max(n1,n2,n3)+20)
savefig(p1, "CoefComp.pdf")