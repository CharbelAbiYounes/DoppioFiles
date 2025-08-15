using LinearAlgebra, Plots, LaTeXStrings, Statistics
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
h = x-> a<x<b ? (x^4+1)*sqrt(x-a)*sqrt(b-x)/x^2 : 0
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
evals = eigvals(W)
true_spikes = evals[end:-1:end-length(spikes)+1]

vecNbr = 100
vecList = randn(N,vecNbr)

# # No Averaging
seq_len = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 2.1/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
TrueChol,ModChol = CholSeq(W,tol,seq_len,jmp,max_iter,vecNbr,vecList=vecList)
γmin, γplus = EstimSupp([TrueChol[1]])
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,[TrueChol[1]],N)
SpikeNbr, SpikeLoc = EstimSpike([TrueChol[1]],N,c=1.0)

p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="No Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p, "DensNoAvrg.pdf")

display("No Average -- Number of Iterations "*string(size(TrueChol[1], 1)))
display("No Average -- Number of Spikes "*string(SpikeNbr))

# Averaging
γmin, γplus = EstimSupp(ModChol)
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,ModChol,N)
SpikeNbr, SpikeLoc = EstimSpike(TrueChol,N,c=1.0)

p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p, "DensAvrg.pdf")

sizelist = zeros(Float64,length(ModChol))
for i=1:length(sizelist)
    sizelist[i] = size(ModChol[i], 1)
end
display("Average -- Number of Iterations "*string(maximum(sizelist)))
display("Average -- Number of Spikes "*string(SpikeNbr))

# Trace Measure (Block Lanczos)
BL_iter = min(max_iter,convert(Int64,floor(N/(5*vecNbr))))
Qk, Q_rem, A, B, B_0 = block_lanczos(W, vecList, BL_iter)
T = block_tridiag(A,B)
T = (T+T')/2
Chol = cholesky(T)
L = Chol.L
n = convert(Int64,size(L,1)/vecNbr)
C = [zeros(Float64,vecNbr,vecNbr) for _ in 1:n]
D = [zeros(Float64,vecNbr,vecNbr) for _ in 1:n-1]
for i=1:n-1
    C[i] = L[(i-1)*vecNbr+1:i*vecNbr,(i-1)*vecNbr+1:i*vecNbr]
    D[i] = L[i*vecNbr+1:(i+1)*vecNbr,(i-1)*vecNbr+1:i*vecNbr]
end
C[n] = L[(n-1)*vecNbr+1:n*vecNbr,(n-1)*vecNbr+1:n*vecNbr]

seq = 3
j = FindSteadyState(C,D,tol/300,min(15,BL_iter),seq)

ModC1, ModD1 = ModChol1(C[1:j+seq+1],D[1:j+seq])
ModC2, ModD2 = ModChol2(C[1:j+seq+1],D[1:j+seq])

x = -0.1:0.001:10
yAvrg = BlockEstimDensity2(x,C[1:j+seq+1],D[1:j+seq],N)
p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Average")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p,"DensBlockLanczos3.pdf")

γmin, γplus = BlockEstimSupp(ModC1,ModD1)
x = (-0.1+γmin):0.001:(γplus+0.1)
yAvrg = BlockEstimDensity1(x,ModC1,ModD1,N)
SpikeNbr, SpikeLoc = BlockEstimSpike(ModC1,ModD1,γplus,N,c=1.0)
p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p,"DensBlockLanczos1.pdf")

display("Trace/BlockLanczos Average 1 -- Number of Spikes "*string(SpikeNbr))

γmin, γplus = BlockEstimSupp(ModC2,ModD2)
x = (-0.1+γmin):0.001:(γplus+0.1)
yAvrg = BlockEstimDensity1(x,ModC2,ModD2,N)
SpikeNbr, SpikeLoc = BlockEstimSpike(ModC2,ModD2,γplus,N,c=1.0)
p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p,"DensBlockLanczos2.pdf")

display("Trace/BlockLanczos Average 2 -- Number of Spikes "*string(SpikeNbr))

n = length(C)
display("Trace/BlockLanczos Average -- Number of Iterations "*string(j+seq+1))

# Tensorized Average
tallvec = vec(vecList)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
γmin, γplus = KronEstimSupp(KronChol)
x = -0.1+γmin:0.001:γplus+0.1
dens = KronEstimDensity(x,KronChol,N::Integer)
SpikeNbr, SpikeLoc = KronEstimSpike(KronChol, γplus, N)

p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,dens,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p, "DensKron.pdf")

n = size(KronChol,1)
display("Kron Average -- Number of Iterations "*string(n))
display("Kron Average -- Number of Spikes "*string(SpikeNbr))

# Alternative Tensorized Average
Vec = copy(vecList)
for i=1:size(Vec,2)
    Vec[:,i] = Vec[:,i]/norm(Vec[:,i])
end
tallvec = vec(Vec)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
γmin, γplus = KronEstimSupp(KronChol)
x = -0.1+γmin:0.001:γplus+0.1
dens = KronEstimDensity(x,KronChol,N::Integer)
SpikeNbr, SpikeLoc = KronEstimSpike(KronChol, γplus, N)

p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,dens,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p, "DensAltKron.pdf")

n = size(KronChol,1)
display("Alt Kron Average -- Number of Iterations "*string(n))
display("Alt Kron Average -- Number of Spikes "*string(SpikeNbr))

# Trace Measure (Kron)
Q, R = qr(vecList)
sgn = Diagonal(sign.(diag(R)))
R = sgn*R
Q = Q*sgn
tallvec = vec(Q)
KronChol = KronConvChol(W,tol,seq_len,jmp,max_iter,tallvec)
γmin, γplus = KronEstimSupp(KronChol)
x = -0.1+γmin:0.001:γplus+0.1
dens = KronEstimDensity(x,KronChol,N::Integer)
SpikeNbr, SpikeLoc = KronEstimSpike(KronChol, γplus, N)

p = histogram(evals,bins=-0.2:0.1:6,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(x,dens,linecolor=:red,linewidth=3,label="Average")
p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p, "DensTraceKron.pdf")

n = size(KronChol,1)
display("Trace Kron Average -- Number of Iterations "*string(n))
display("Trace Kron Average -- Number of Spikes "*string(SpikeNbr))