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

function evaluateCollocation(self::CollocationFunction, z::Float64)
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

function solveCollocation(self::CollocationFunction, strike::Float64)
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

    allck = PolynomialRoots.roots([cc,bb,aa])
    for ck in allck
        if abs(imag(ck)) < 1e-10 && real(ck) >= self.x[i-1]-1e-10 && real(ck) <= self.x[i]+1e-10
            return real(ck)
        end
    end
    println("ERRROR !!! no roots found in range", allck," ", strike, " ",aa, bb, cc," ", i," ",self.x[i-1]," ",self.x[i])
    return self.x[1]
end

function phi(R, cx,cv, dx, dv)
    lenr = nrows(dx)
    lenc = ncols(dx)
    cx = R(cx)
    cv = R(cv)
    phi = zero_matrix(R, lenr,lenc)
    dphidx = zero_matrix(R, lenr,lenc)
    d2phidx2 = zero_matrix(R, lenr,lenc)
    dphidv = zero_matrix(R, lenr,lenc)
    d2phidv2 = zero_matrix(R, lenr,lenc)
    d2phidxdv  =zero_matrix(R, lenr,lenc)
    for i=1:lenr
        for j=1:lenc
            sqx = (dx[i,j] / cx)^2
            sqv = (dv[i,j] / cv)^2
            phi[i,j] = sqrt(1 + sqx+sqv)
            dphidx[i,j] = (dx[i,j] / (cx^2)) / phi[i,j]
            phi3 = phi[i,j] ^3
            d2phidx2[i,j] = (R(1.0) / (cx^2)) / phi[i,j] - (sqx / (cx^2)) / phi3
            dphidv[i,j] = (dv[i,j] / (cv^2)) / phi[i,j]
            d2phidv2[i,j] = (R(1.0) / (cv^2)) / phi[i,j] - (sqv / (cv^2)) / phi3
            d2phidxdv[i,j] = -((dx[i,j] / (cx^2)) * (dv[i,j] / (cv^2))) / phi3
        end
    end
    return phi, dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv
end

function makeRBFMatrix(R, PX,PV,cx,cv)
    len = nrows(PX)
    rx = zero_matrix(R, len,len)
    rv = zero_matrix(R, len,len)
    for i = 1:len
        for j = 1:len
            rx[j,i] = PX[j,1] - PX[i,1]
            rv[j,i] = PV[j,1] - PV[i,1]
        end
    end
    I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = phi(R, cx,cv,rx,rv)
    return I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV
end

function approx_lu!(P::Generic.perm, luMat::arb_mat, x::arb_mat)
    # int arb_mat_approx_lu(slong * P, arb_mat_t LU, const arb_mat_t A, slong prec)
    # int arb_mat_approx_solve(arb_mat_t X, const arb_mat_t A, const arb_mat_t B, slong prec)
    r = ccall((:arb_mat_approx_lu, :libarb), Cint,
    # r = ccall((:arb_mat_lu, :libarb), Cint,
    (Ptr{Int}, Ref{arb_mat},  Ref{arb_mat}, Int),
    P.d, luMat, x, prec(base_ring(x)))
    r == 0 && error("Could not find $(nrows(x)) invertible pivot elements")
    P.d .+= 1
    return nrows(x)
end

function approx_solve_lu_precomp!(z::arb_mat, P::Generic.perm, luMat::arb_mat, y::arb_mat)
    # r = ccall((:arb_mat_solve_lu_precomp, :libarb), Nothing,
    r = ccall((:arb_mat_approx_solve_lu_precomp, :libarb), Nothing,
    (Ref{arb_mat}, Ptr{Int}, Ref{arb_mat},  Ref{arb_mat}, Int),
    z, P.d .- 1, luMat, y, prec(base_ring(y)))
    nothing
end
#void arb_mat_approx_solve_lu_precomp(arb_mat_t X, const slong * perm, const arb_mat_t A, const arb_mat_t B, slong prec)

function approx_inv(x::arb_mat)
    y = one(parent(x))
    approx_solve!(y, x, y)
    return y
end

function approx_solve!(x::arb_mat)
    r = ccall((:arb_mat_approx_solve, :libarb), Cint,
    (Ref{arb_mat}, Ref{arb_mat}, Ref{arb_mat}, Int),
    z, x, y, prec(base_ring(x)))
    r == 0 && error("Matrix cannot be inverted numerically")
    approx_solve!(y, x, y)
    return y
end

function priceR(R, xFactor, vFactor, isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
    epsilon = 1e-3
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*exp(-kappa*T)/(sigma*sigma*(1-exp(-kappa*T)))
    ncx2 = NoncentralChisq(dChi,v0*chiN)
    vmax = quantile(ncx2,1-epsilon)*exp(-kappa*T)/chiN
    vmin = quantile(ncx2,epsilon)*exp(-kappa*T)/chiN
    vmin = max(vmin,1e-4)
    V = collect(range(vmin,stop=vmax,length=L))
    Xspan = 6*sqrt(theta*T)
    logK = log(K)
    Kinv = solveCollocation(cFunc,K)
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
        Sc[i] = evaluateCollocation(cFunc,S[i])
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
    PX = zero_matrix(R, L*M,1)
    PV = zero_matrix(R, L*M,1)
    for j = 1:L
        for i = 1:M
            PX[i+(j-1)*M,1]= X[i]
            PV[i+(j-1)*M,1] = V[j]
        end
    end
    Iphi,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = makeRBFMatrix(R, PX,PV,cx,cv)
    dt = -T/N
    A = zero_matrix(R,L*M,L*M)
    hdt = R(0.5*dt)
    dtrq = R(dt*(r-q))
    rdt = R(r*dt)
    sig2 = R(sigma^2)
    dtkappa = R(dt*kappa)
    thetar = R(theta)
    dtrs = R(dt*rho*sigma)
    for i = 1:L*M
        for j=1:L*M
            pv = PV[i,1]
            dtpv = hdt*pv
            a1 = dtpv * d2IdX2[i,j] + (dtrq - dtpv)*dIdX[i,j]-rdt*Iphi[i,j]
            a2 = sig2 * dtpv * d2IdV2[i,j] + (dtkappa * (thetar-pv)) * dIdV[i,j] + (dtrs*pv)*d2IdXdV[i,j]
            A[i,j] = a1+a2
        end
    end
    etime = @elapsed begin
        for j = 1:L
            for i =1:M*L
                A[1+(j-1)*M,i] = 0
                A[M+(j-1)*M,i] = 0
            end
        end
        a = R(1.0 - sqrt(2.0)/2.0)
        reg = 0e-15

        # IphiR = matrix(R, [R(Iphi[i,j]) for i in 1:L*M, j in 1:L*M])
        # AR = matrix(R, [R(A[i,j]) for i in 1:L*M, j in 1:L*M])
        FR = zero_matrix(R, L*M, 1)
        for i = 1:L*M
            FR[i,1] = R(F[i])
        end
        coeff = similar(FR)
        # IphiInv = approx_inv(Iphi)
        # B = A # Ip * inv(Ip + a*A) = inv (I + a*A*inv(Ip))
        # B *= a
        # # for i=1:L*M
        # #     B[i,i] += 1
        # # end
        # B += Iphi
        # LiInv = approx_inv(B)
        # LiInv = Iphi*LiInv
        P0 = perm(collect(1:L*M))
        luMat0 = similar(Iphi)
        lu0 = approx_lu!(P0, luMat0, Iphi)
        Li = Iphi + a * A
        Pi = perm(collect(1:L*M))
        luMati = similar(Li)
        lui = approx_lu!(Pi, luMati, Li)

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
            approx_solve_lu_precomp!(coeff, Pi, luMati, FR)
            F1 = Iphi*coeff
            # F1 = LiInv * FR
            for j = 1:L
                if isCall
                    F1[1+(j-1)*M,1] = 0
                    F1[M+(j-1)*M,1] =  max(Sc[M]*exp(-q*ti)-K*exp(-r*ti),0.0)
                else
                    F1[1+(j-1)*M,1] = max(K*exp(-r*ti)-Sc[1]*exp(-q*ti),0.0)
                    F1[M+(j-1)*M,1] = 0.0
                end
            end
            approx_solve_lu_precomp!(coeff, Pi, luMati, F1)
            F2 = Iphi*coeff
            # F2 = LiInv*F1
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
    approx_solve_lu_precomp!(coeff, P0, luMat0, FR)
    coeffR = coeff
    # coeffR = IphiInv * FR
    # coeff=[Float64(coeffR[i,1]) for i in 1:L*M]
    # println("coeff",coeff)
    for i = 1:length(spotArray)
        spot = spotArray[i]
        refPrice = priceArray[i]
        logspot = log(spot)
        lenr = nrows(PX)
        px0 = zero_matrix(R,lenr,ncols(PX))
        pv0 = zero_matrix(R,nrows(PV),ncols(PV))
        for j=1:lenr
            px0[j,1] = logspot - PX[j,1]
            pv0[j,1] = v0 - PV[j,1]
        end
        phispotv0,dphi,d2phi,dphidv,d2phidv2, dphidxdv = phi(R, cx,cv,px0,pv0)
        price = (transpose(phispotv0) * coeffR)[1,1]
        delta = (transpose(dphi) * coeffR)[1,1]/R(spot)
        gamma = (transpose(d2phi) * coeffR)[1,1]/R(spot^2)
        error = price - R(refPrice)
        println(K," ",cx," ",cv," ",xFactor," ",vFactor," ARB ",N, " ",M," ", L," ", Float64(price)," ", Float64(delta)," ",Float64(gamma)," ",Float64(error), " ",etime)
    end
end

A=[0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941]
C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
leftSlope=0.8329310535215612
rightSlope=0.2668764075068484
cFunc = CollocationFunction(A,B,C,X,leftSlope,rightSlope)

kappa = 0.35
theta = 0.321
sigma = 1.388
rho = -0.63
r = 0.0
q = 0.0
v0=0.133
T=0.4986301369863014
K=1.0
M=30
L=30
N=32
spotArray = [1.0]
priceArray= [0.07278065]
isCall = false
xFactor=2.0
vFactor=2.0
R = ArbField(256)
priceR(R,xFactor,vFactor,isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
xFactors = [1+Float64(10*i)/40 for i in 0:40]
#xFactors = [11+Float64(10*i)/40 for i in 1:10]
xFactors = reverse(xFactors)
for xFactor in xFactors
    vFactor = xFactor
    priceR(R, xFactor,vFactor,isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)
end
