using Nemo
using Distributions
using LinearAlgebra
using PolynomialRoots


struct CollocationFunction
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    x::Array{Float64}
    leftSlope::Float64
    rightSlope::Float64
end

function evaluate(self::CollocationFunction, z::Float64)
    if z <= self.x[1]
        return self.leftSlope*(z-self.x[1]) + self.a[1]
    elseif z >= self.x[end]
        return self.rightSlope*(z-self.x[end])+self.a[end]
    end
    i = searchsortedfirst(self.x,z)  # x[i-1]<z<=x[i]
    if i > 1
        i -= 1
    end
    h = z-self.x[i]
    return self.a[i] + h*(self.b[i]+h*self.c[i])
end

function solve(self::CollocationFunction, strike::Float64)
    if strike <= self.a[1]
        sn = self.leftSlope
        return (strike-self.a[1])/sn + self.x[1]
    elseif strike >= self.a[end]
        sn = self.rightSlope
        return (strike-self.a[end])/sn + self.x[end]
    end
    i = searchsortedfirst(self.a,strike)  # a[i-1]<strike<=a[i]
    if abs(self.a[i]-strike)< 1e-10
        return self.x[i]
    elseif abs(self.a[i-1]-strike)< 1e-10
        return self.x[i-1]
    elseif i == 1
        i += 1
    end
    x0 = self.x[i-1]
    c = self.c[i-1]
    b = self.b[i-1]
    a = self.a[i-1]
    d = 0
    cc = a + x0*(-b+x0*(c-d*x0)) - strike
    bb = b + x0*(-2*c+x0*3*d)
    aa = -3*d*x0 + c

    allck = roots([cc,bb,aa])
    for ck in allck
        if abs(imag(ck)) < 1e-10 && real(ck) >= self.x[i-1]-1e-10 && real(ck) <= self.x[i]+1e-10
            return real(ck)
        end
    end
    println("ERRROR !!! no roots found in range", allck," ", strike, " ",aa, bb, cc," ", i," ",self.x[i-1]," ",self.x[i])
    return self.x[1]
end

function phi(cx,cv, dx, dv)
    sqx = (dx ./ cx).^2
    sqv = (dv ./ cv).^2
    phi = sqrt.(1 .+ sqx+sqv)
    dphidx = (dx ./ (cx .* cx)) ./ phi
    phi3 = phi .^3
    d2phidx2 = (1.0 ./ (cx .* cx)) ./ phi - (sqx ./ (cx .* cx)) ./ phi3
    dphidv = (dv ./ (cv .* cv)) ./ phi
    d2phidv2 = (1.0 ./ (cv .* cv)) ./ phi - (sqv ./ (cv .* cv)) ./ phi3
    d2phidxdv = -((dx ./ (cx .* cx)) .* (dv ./ (cv .* cv))) ./ phi3
    return phi, dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv
end

function makeRBFMatrix(PX,PV,cx,cv)
    len = length(PX)
    rx = zeros(typeof(PX[1,1]),len,len)
    rv = zeros(typeof(PX[1,1]),len,len)
    for i = 1:len
        rx[:,i] = PX .- PX[i]
        rv[:,i] = PV .- PV[i]
    end
    I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = phi(cx,cv,rx,rv)
    return I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV
end

function price(isCall, xFactor, vFactor, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
    epsilon = 1e-4
    boundaryCondition = "Dirichlet" #Dirichlet, ZeroGamma, Gamma
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*exp(-kappa*T)/(sigma*sigma*(1-exp(-kappa*T)))
    ncx2 = NoncentralChisq(dChi,v0*chiN)
    vmax = quantile(ncx2,1-epsilon)*exp(-kappa*T)/chiN
    vmin = quantile(ncx2,epsilon)*exp(-kappa*T)/chiN
    vmin = max(vmin,1e-4)
    V = collect(range(vmin,stop=vmax,length=L))
    Xspan = 4*sqrt(theta*T)
    logK = log(K)
    Kinv = solve(cFunc,K)
    logKinv = log(Kinv)
    Xmin = logK - Xspan + (r-q)*T - 0.5*v0*T
    Xmax = logK + Xspan + (r-q)*T - 0.5*v0*T
    X = collect(range(Xmin,stop=Xmax,length=M))
    hm = X[2]-X[1]
    S = exp.(X)
    cx = xFactor*(Xmax-Xmin)/(M-1)
    cv = vFactor*(vmax-vmin)/(L-1)
    Sc = zeros(M)
    for i = 1:M
        Sc[i] = evaluate(cFunc,S[i])
    end
    if isCall
        F0 = max.(Sc .- K, 0.0)
    else
        F0 = max.(K .- Sc, 0.0)
    end
    F0smooth = F0
    F = zeros(L*M)
    for j = 1:L
        F[1+(j-1)*M:j*M] = F0smooth
    end
    PX = zeros(Float64,L*M)
    PV = zeros(typeof(PX[1,1]),L*M)
    for j = 1:L
        PX[1+(j-1)*M:j*M]=X
        PV[1+(j-1)*M:j*M] .= V[j]
    end
    Iphi,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = makeRBFMatrix(PX,PV,cx,cv)
    dt = -T/N
    PV2D = zeros(typeof(PX[1,1]),L*M,L*M)
    for j = 1:L*M
        PV2D[:,j] = PV
    end
    dtPV = (0.5*dt) .* PV2D
    A1 = dtPV .* d2IdX2 + (dt*(r-q) .- dtPV) .* dIdX - (r*dt) .* Iphi
    A2 = (sigma^2 .* dtPV) .* d2IdV2 + (dt*kappa .* (theta .- PV2D)) .* dIdV + (dt*rho*sigma .* PV2D) .* d2IdXdV
    A = A1+A2
    etime = @elapsed begin
        #dirichlet
        if boundaryCondition == "Dirichlet"
            for j = 1:L
                A[1+(j-1)*M,:] .= 0
                A[M+(j-1)*M,:] .= 0
            end
        elseif  boundaryCondition == "ZeroGamma"
            #second = 0
            for j = 1:L
                A[1+(j-1)*M,:] .= d2IdX2[1+(j-1)*M,:] .- dIdX[1+(j-1)*M,:]
                A[M+(j-1)*M,:] .= d2IdX2[M+(j-1)*M,:] .- dIdX[M+(j-1)*M,:]
            end
        elseif  boundaryCondition == "Gamma"
            #second = 0
            for j = 1:L
                A[1+(j-1)*M,:] .= (dt*(r-q) ) .* dIdX[1+(j-1)*M,:] - (r*dt) .* Iphi[1+(j-1)*M,:]
                A[M+(j-1)*M,:] .=(dt*(r-q) ) .* dIdX[M+(j-1)*M,:] - (r*dt) .* Iphi[M+(j-1)*M,:]
            end
        end

        reg = 0e-15
        # IphiR = matrix(R, [R(Iphi[i,j]) for i in 1:L*M, j in 1:L*M])
        # AR = matrix(R, [R(A[i,j]) for i in 1:L*M, j in 1:L*M])
        # Iphi = convert(Array{BigFloat},Iphi)
        # A = convert(Array{BigFloat},A)
        lu0 = factorize(Iphi+reg*LinearAlgebra.I)
        a = 1.0 - sqrt(2.0)/2.0
        Li = Iphi + a .* A
        if boundaryCondition == "ZeroGamma"
            for j = 1:L
                Li[1+(j-1)*M,:] .= A[1+(j-1)*M,:]
                Li[M+(j-1)*M,:] .= A[M+(j-1)*M,:]
            end
        end
        lui = factorize(Li)
        # println("Condition Iphi ",cond(Iphi), " A ",cond(Li))
        # d0 = diag(lu0.U)
        # di = diag(lui.U)
        # println("sq ratio ", maximum(d0 .* d0)/minimum(d0 .* d0)," A ", maximum(di .* di)/minimum(di .* di))
        for i = 1:N
            ti = (T*i)/N
            for j = 1:L
                if boundaryCondition == "Dirichlet"
                    if isCall
                        F[1+(j-1)*M] = 0
                        F[M+(j-1)*M] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                    else
                        F[1+(j-1)*M] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                        F[M+(j-1)*M] = 0.0
                    end
                elseif  boundaryCondition == "ZeroGamma"
                    F[1+(j-1)*M] = 0
                    F[M+(j-1)*M] = 0
                end
            end
            coeff = lui \ F
            F1 = Iphi*coeff
            for j = 1:L
                if boundaryCondition == "Dirichlet"

                    if isCall
                        F1[1+(j-1)*M] = 0
                        F1[M+(j-1)*M] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                    else
                        F1[1+(j-1)*M] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                        F1[M+(j-1)*M] = 0.0
                    end
                elseif  boundaryCondition == "ZeroGamma"
                    F1[1+(j-1)*M] = 0
                    F1[M+(j-1)*M] = 0
                end
            end
            coeff = lui \ F1
            F2 = Iphi*coeff
            for j = 1:L
                if boundaryCondition == "Dirichlet"
                    if isCall
                        F2[1+(j-1)*M] = 0
                        F2[M+(j-1)*M] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                    else
                        F2[1+(j-1)*M] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                        F2[M+(j-1)*M] = 0.0
                    end
                elseif  boundaryCondition == "ZeroGamma"
                    F2[1+(j-1)*M] = 0
                    F2[M+(j-1)*M] = 0
                end
            end
            F = (1+sqrt(2.0)) .* F2 - sqrt(2.0) .* F1
        end
    end  #elapsed
    coeff = lu0 \ F

    for i = 1:length(spotArray)
        spot = spotArray[i]
        refPrice = priceArray[i]
        logspot = log(spot)
        phispotv0,dphi,d2phi,dphidv,d2phidv2, dphidxdv = phi(cx,cv,logspot  .- PX,v0 .- PV)
        price = dot(phispotv0,  coeff)
        delta = dot(dphi, coeff)/spot
        gamma = dot(d2phi, coeff)/(spot^2)-delta/spot
        error = price -refPrice
        println(K," ",cx," ",cv," ",xFactor," ",vFactor," Float64 ",N, " ",M," ", L," ", Float64(price)," ", Float64(delta)," ",Float64(gamma)," ",Float64(error), " ",etime)
    end

    for spot in S
        logspot = log(spot)
        phispotv0,dphi,d2phi,dphidv,d2phidv2, dphidxdv = phi(cx,cv,logspot  .- PX,v0 .- PV)
        price = dot(phispotv0,  coeff)
        delta = dot(dphi, coeff)/spot
        gamma = dot(d2phi, coeff)/(spot^2)-delta/spot
        println(spot," LS ",N, " ",M," ", L," ", price," ", delta," ",gamma)
    end
end

function approx_solve!(z::arb_mat, x::arb_mat, y::arb_mat)
    r = ccall((:arb_mat_approx_solve, :libarb), Cint,
    (Ref{arb_mat}, Ref{arb_mat}, Ref{arb_mat}, Int),
    z, x, y, prec(base_ring(x)))
    r == 0 && error("Matrix cannot be inverted numerically")
    nothing
end

function approx_inv(x::arb_mat)
    y = one(parent(x))
    approx_solve!(y, x, y)
    return y
end
function priceInv(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
    epsilon = 1e-3
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*exp(-kappa*T)/(sigma*sigma*(1-exp(-kappa*T)))
    ncx2 = NoncentralChisq(dChi,v0*chiN)
    vmax = quantile(ncx2,1-epsilon)*exp(-kappa*T)/chiN
    vmin = quantile(ncx2,epsilon)*exp(-kappa*T)/chiN
    vmin = max(vmin,1e-4)
    V = collect(range(vmin,stop=vmax,length=L))
    Xspan = 8*sqrt(theta*T)
    logK = log(K)
    Kinv = solve(cFunc,K)
    logKinv = log(Kinv)
    Xmin = logK*0 - Xspan + (r-q)*T - 0.5*v0*T
    Xmax = logK*0 + Xspan + (r-q)*T - 0.5*v0*T
    X = collect(range(Xmin,stop=Xmax,length=M))
    hm = X[2]-X[1]
    S = exp.(X)
    cx = 5*(Xmax-Xmin)/(M-1)
    cv = 10*(vmax-vmin)/(L-1)
    Sc = zeros(M)
    for i = 1:M
        Sc[i] = evaluate(cFunc,S[i])
    end
    if isCall
        F0 = max.(Sc .- K,0)
    else
        F0 = max.(K .- Sc,0)
    end
    F0smooth = F0
    F = zeros(L*M)
    for j = 1:L
        F[1+(j-1)*M:j*M] = F0smooth
    end
    PX = zeros(Float64,L*M)
    PV = zeros(typeof(PX[1,1]),L*M)
    for j = 1:L
        PX[1+(j-1)*M:j*M]=X
        PV[1+(j-1)*M:j*M] .= V[j]
    end
    Iphi,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = makeRBFMatrix(PX,PV,cx,cv)
    dt = -T/N
    PV2D = zeros(typeof(PX[1,1]),L*M,L*M)
    for j = 1:L*M
        PV2D[:,j] = PV
    end
    dtPV = (0.5*dt) .* PV2D
    A1 = dtPV .* d2IdX2 + (dt*(r-q) .- dtPV) .* dIdX - (r*dt) .* Iphi
    A2 = (sigma^2 .* dtPV) .* d2IdV2 + (dt*kappa .* (theta .- PV2D)) .* dIdV + (dt*rho*sigma .* PV2D) .* d2IdXdV
    A = A1+A2
    etime = @elapsed begin
        for j = 1:L
            A[1+(j-1)*M,:] .= 0
            A[M+(j-1)*M,:] .= 0
        end
        a = 1.0 - sqrt(2.0)/2.0
        reg = 0e-15
        # Iphi = convert(Array{BigFloat},Iphi)
        # A = convert(Array{BigFloat},A)
        # lu0 = factorize(Iphi+reg*LinearAlgebra.I)
        # IphiInv = inv(Iphi)
        # # Li = Iphi + a .* A
        # # lui = factorize(Li)
        # LiInv = inv(LinearAlgebra.I + a .* (A*IphiInv))

        IphiR = matrix(R, [R(Iphi[i,j]) for i in 1:L*M, j in 1:L*M])
        AR = matrix(R, [R(A[i,j]) for i in 1:L*M, j in 1:L*M])
        FR = zero_matrix(R, L*M, 1)
        for i = 1:L*M
            FR[i,1] = R(F[i])
        end

        IphiInv = approx_inv(IphiR)
        B = AR # Ip * inv(Ip + a*A) = inv (I + a*A*inv(Ip))
        B *= R(a)
        # for i=1:L*M
        #     B[i,i] += 1
        # end
        B += IphiR
        LiInv = approx_inv(B)
        LiInv = IphiR*LiInv
        for i = 1:N
            ti = (T*i)/N
            for j = 1:L
                if isCall
                    FR[1+(j-1)*M,1] = 0
                    FR[M+(j-1)*M,1] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                else
                    FR[1+(j-1)*M,1] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                    FR[M+(j-1)*M,1] = 0.0
                end
            end
            #coeff = LiInv * F
            #F1 = Iphi*coeff
            F1 = LiInv * FR
            for j = 1:L
                if isCall
                    F1[1+(j-1)*M,1] = 0
                    F1[M+(j-1)*M,1] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                else
                    F1[1+(j-1)*M,1] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                    F1[M+(j-1)*M,1] = 0.0
                end
            end
            # coeff = LiInv * F1
            # F2 = Iphi*coeff
            F2 = LiInv*F1
            for j = 1:L
                if isCall
                    F2[1+(j-1)*M,1] = 0
                    F2[M+(j-1)*M,1] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                else
                    F2[1+(j-1)*M,1] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                    F2[M+(j-1)*M,1] = 0.0
                end
            end
            for index=1:L*M
                FR[index,1] = (1+sqrt(2.0)) * F2[index,1] - sqrt(2.0)*F1[index,1]
            end
        end
    end  #elapsed
    coeffR = IphiInv * FR
    coeff=[Float64(coeffR[i,1]) for i in 1:L*M]
    println("coeff",coeff)
    for i = 1:length(spotArray)
        spot = spotArray[i]
        refPrice = priceArray[i]
        logspot = log(spot)
        phispotv0,dphi,d2phi,dphidv,d2phidv2, dphidxdv = phi(cx,cv,logspot  .- PX,v0 .- PV)
        price = dot(phispotv0,  coeff)
        delta = dot(dphi, coeff)/spot
        gamma = dot(phispotv0, coeff)/(spot^2)
        error = price -refPrice
        println(spot," LS ",N, " ",M," ", L," ", price," ", delta," ",gamma," ",error, " in ",etime)
    end
end

A=[0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941]
C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
leftSlope=0.8329310535215612
rightSlope=0.2668764075068484
cFunc = CollocationFunction(A,B,C,X,leftSlope,rightSlope)
A = X
leftSlope = 1.0
rightSlope = 1.0
C = 0*C
B = [1.0 for i=1:length(B)]
kappa = 0.35
theta = 0.321
sigma = 1.388
rho = -0.63
r = 0.0
q = 0.0
v0=0.133
T=0.4986301369863014
M=30
L=30
N=32
spotArray = [1.0]
K=1.0
priceArray= [0.07278065]
K=0.7
priceArray = [0.00960629]
K=1.4
priceArray = [0.40015225]
K=0.8
priceArray = [0.01942598]
K=1.2
priceArray = [0.20760077]
isCall = false
xFactor=2.0
vFactor=2.0
xFactors = [1+Float64(10*i)/40 for i in 0:40]
xFactors = reverse(xFactors)
for xFactor in xFactors
    vFactor = xFactor
    price(isCall,xFactor,vFactor, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
end
