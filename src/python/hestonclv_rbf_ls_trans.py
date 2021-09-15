import numpy as np
import math
import time
from scipy.sparse import csc_matrix, lil_matrix, dia_matrix, identity, linalg as sla
from scipy import linalg as la
from scipy.stats import ncx2, norm
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numba
from numba import jit

v0=0.05412
theta=0.04
sigma=0.3
kappa=1.5
rho=-0.9
r=0.02
q=0.05
T=0.15
refPrice=4.108362515  #rouah
refPrice = 8.89486909 #albrecher

def phi(cx,cv, dx, dv):
    # return phiGaussian(cx,cv, dx, dv)
    return phiMultiquadric(cx,cv, dx, dv)

def phiGaussian(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    phi = np.exp(-sqx - sqv)
    dphidx = np.multiply(-2*(dx)/(cx*cx), phi)
    d2phidx2 = np.multiply((4*sqx-2)/(cx*cx),phi)
    dphidv = np.multiply(-2*(dv)/(cv*cv), phi)
    d2phidv2 = np.multiply((4*sqv-2)/(cv*cv),phi)
    d2phidxdv = np.multiply(np.multiply(-2*(dx)/(cx*cx),-2*(dv)/(cv*cv)), phi)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def phiW(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    r2 = sqx+sqv
    r2 = np.maximum(r2,1e-4)
    r = np.sqrt(r2)
    oner6 = np.power(np.maximum(1-r,0),6)
    quad = 35.0*r2+18*r+3
    phi = np.multiply(oner6,quad)
    oner5 = np.power(np.maximum(1-r,0),5)
    dphidr = np.multiply(oner6,70*r+18)-6*np.multiply(oner5,quad)
    oner4 = np.power(np.maximum(1-r,0),4)
    d2phidr2 = np.multiply(30*oner4,quad)-12*np.multiply(oner5, 70*r+18)+70*oner6
    xratio = np.divide(dx,r*cx*cx)
    vratio = np.divide(dv, r*cv*cv)
    dphidx = np.multiply(dphidr,xratio)
    dphidv = np.multiply(dphidr,vratio)
    r3 = np.power(r,3)
    x2ratio = np.divide(r2*cx*cx-sqx,cx*cx*cx*cx*r3)
    d2phidx2 = np.multiply(d2phidr2,np.square(xratio))+ np.multiply(dphidr,x2ratio)
    v2ratio = np.divide(r2*cv*cv-sqv,cv*cv*cv*cv*r3)
    xvratio = -np.divide(np.multiply(dx,dv),cx*cx*cv*cv*r3)
    d2phidv2 = np.multiply(d2phidr2,np.square(vratio))+ np.multiply(dphidr,v2ratio)
    d2phidxdv = np.multiply(dphidr, xvratio)+np.multiply(d2phidr2, np.multiply(xratio,vratio))
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv


def phiMultiquadric(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    phi = np.sqrt(1+sqx+sqv)
    dphidx = np.divide(dx/(cx*cx),phi)
    phi3 = np.power(phi,3)
    d2phidx2 = np.divide(1.0/(cx*cx),phi)-np.divide(sqx/(cx*cx),phi3)
    dphidv = np.divide(dv/(cv*cv),phi)
    d2phidv2 = np.divide(1.0/(cv*cv),phi)-np.divide(sqv/(cv*cv),phi3)
    d2phidxdv = -np.divide(np.multiply(dx/(cx*cx),dv/(cv*cv)),phi3)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def phiIM2(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    r2= 1+sqx+sqv
    phi = np.divide(1.0,r2)
    phi2 = np.power(phi,2)
    phi3 = np.power(phi,3)
    dphidx = -np.multiply(2*dx/(cx*cx),phi2)
    d2phidx2 = np.multiply(8*sqx/(cx*cx),phi3)-np.multiply(2/(cx*cx),phi2)
    dphidv = -np.multiply(2*dv/(cv*cv),phi2)
    d2phidv2 = np.multiply(8*sqv/(cv*cv),phi3)-np.multiply(2/(cv*cv),phi2)
    d2phidxdv = np.multiply(8*np.multiply(dx/(cx*cx),dv/(cv*cv)),phi3)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def phiIM(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    sqrt = np.sqrt(1+sqx+sqv)
    phi = np.divide(1.0,sqrt)
    phi3 = np.power(phi,3)
    dphidx = -np.multiply(dx/(cx*cx),phi3)
    phi5 = np.power(phi,5)
    d2phidx2 = np.multiply(sqx/(cx*cx),phi5)-np.multiply(1/(cx*cx),phi3)
    dphidv = -np.multiply(dv/(cv*cv),phi3)
    d2phidv2 = np.multiply(sqv/(cv*cv),phi5)-np.multiply(1/(cv*cv),phi3)
    d2phidxdv = np.multiply(3*np.multiply(dx/(cx*cx),dv/(cv*cv)),phi5)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def phiTPS2(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    r2 = sqx+sqv
    r2 = np.maximum(1e-8,r2)
    logxv = np.log(r2)
    phi = np.multiply(np.square(r2),logxv)
    x3 = np.multiply(dx/(cx*cx),r2)
    dphidx = np.multiply(x3,4*logxv+2)
    d2phidx2 = np.multiply(r2/(cx*cx),logxv*4+2)+np.multiply(sqx/(cx*cx),8*logxv+12)
    v3 = np.multiply(dv/(cv*cv),r2)
    dphidv =  np.multiply(v3,4*logxv+2)
    d2phidv2 = np.multiply(r2/(cv*cv),logxv*4+2)+np.multiply(sqv/(cv*cv),8*logxv+12)
    d2phidxdv = np.multiply(np.multiply(dx/(cx*cx),dv/(cv*cv)),12+8*logxv)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def phiTPS(cx,cv, dx, dv):
    sqx = np.square(dx/cx)
    sqv = np.square(dv/cv)
    r2 = sqx+sqv
    r2 = np.maximum(1e-16,r2)
    logxv = np.log(r2)
    phi = 0.5*np.multiply(r2,logxv)
    dphidx = np.multiply(dx/(cx*cx),logxv+1)
    d2phidx2 = logxv/(cx*cx) + 1.0/(cx*cx) + np.divide(2*sqx/(cx*cx),r2)
    dphidv =  np.multiply(dv/(cv*cv),logxv+1)
    d2phidv2 = logxv/(cv*cv) + 1.0/(cv*cv) + np.divide(2*sqv/(cv*cv),r2)
    d2phidxdv = 2*np.divide(np.multiply(dx/(cx*cx),dv/(cv*cv)),r2)
    return phi,dphidx, d2phidx2, dphidv, d2phidv2, d2phidxdv

def makeRBFMatrix(PX,PV,cx,cv):
  #use numba loop  i1j1 i2j2 symetric np.zeros((L*M,L*M)) # now I is the collocation matrix.  phi(Pi1j1-Pi2j2)
  # compute (xi-xj)^2/cx^2 + (vi-vj)^2/cv^2
  rx = np.subtract.outer(PX,PX)
  rv = np.subtract.outer(PV,PV)
  # dphi/dr * dr/dx + dphi/dr *dr/dv
  I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = phi(cx,cv,rx,rv)

  return I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV

def priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L):
    isCall = False
    reg = 3e-15
    method = "LS" # "LS","CN","DO"
    smoothing = "None" #,"Averaging","None"
    useDamping = False
    epsilon = 1e-3
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*math.exp(-kappa*T)/(sigma*sigma*(1-math.exp(-kappa*T)))
    vmax = ncx2.ppf((1-epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    vmin = ncx2.ppf((epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    #print("vmax",vmin,vmax, 10*v0)
    #vmax=10.0*v0
    #vmin = 0
    #vmax = 2*v0
    vmin = max(vmin,1e-4)
    #vmax = min(vmax, 10*v0)
    #vmin=0
    V = np.linspace(vmin,vmax,L)
    # W = V
    # hl = W[1]-W[0]
    # JV=np.ones(L)
    # JVm=np.ones(L)

    Xspan = 8*math.sqrt(theta*T)
    logK = math.log(K) #f(e^zi) = K
    Kinv = cFunc.solve(K)
    logKinv = math.log(Kinv)
    Xmin = logK*0 - Xspan + (r-q)*T - 0.5*v0*T
    Xmax = logK*0 + Xspan + (r-q)*T - 0.5*v0*T
    X = np.linspace(Xmin,Xmax,M)
    hm = X[1]-X[0]
    S = np.exp(X)
    cx = 10*(Xmax-Xmin)/(M-1)  #i divide M/2, I multiply cx by 2
    cv = 10*(vmax-vmin)/(L-1)
    # cx = np.random.random_sample((L*M,))*(cx*1.5 - cx*0.75)+cx*0.75
    # cv = np.random.random_sample((L*M,))*(cv*1.5 - cv*0.75)+cv*0.75

    Sc = np.array([cFunc.evaluate(Si) for Si in S])
    if isCall:
        F0 = np.maximum(Sc-K,0)
    else:
        F0 = np.maximum(K-Sc,0)
    F0smooth = np.array(F0,copy=True)

    dIndices = set()
    alldisc = cFunc.X + [Kinv]
    for xd in (alldisc):
        logxd = math.log(xd)
        ixd = np.searchsorted(X,logxd)  # S[i-1]<K<=S[i]
        dIndices.add(ixd)
        if ixd > 0:
            dIndices.add(ixd-1)
    #indices = range(M)

    #print(K, Kinv, cFunc.evaluate(Kinv)-K)
    #raise Error
    if smoothing == "Averaging":
        iStrike = np.searchsorted(X,logKinv)  # S[i-1]<K<=S[i]
        if logKinv < (X[iStrike]+X[iStrike-1])/2:
            iStrike -= 1
        payoff1 = lambda v: cFunc.evaluate(math.exp(v))-K
        payoff1 = np.vectorize(payoff1)
        value = 0
        if isCall:
            a = (X[iStrike]+X[iStrike+1])/2
            value = integrate.quad( payoff1, logKinv, a)
        else:
            a = (X[iStrike]+X[iStrike-1])/2   # int a,lnK K-eX dX = K(a-lnK)+ea-K
            value = integrate.quad( payoff1, logKinv, a)
        h = (X[iStrike+1]-X[iStrike-1])/2
        F0smooth[iStrike] = value[0]/h

    elif smoothing == "Kreiss":
        iStrike = np.searchsorted(X,logKinv)  # S[i-1]<K<=S[i]
        xmk = X[iStrike]
        h = (X[iStrike+1]-X[iStrike-1])/2
        payoff1 = lambda v: (cFunc.evaluate(math.exp(xmk-v))-K)*(1-abs(v)/h)
        payoff1 = np.vectorize(payoff1)
        value = F0smooth[iStrike]
        if isCall:
            a = (X[iStrike]+X[iStrike+1])/2
            #logKinv>0
            value1 = integrate.quad( payoff1, 0,xmk-logKinv)
            value0 = integrate.quad( payoff1, -h, 0)
            value = (value0[0]+value1[0]) /h

        F0smooth[iStrike] = value
        iStrike -= 1
        xmk = X[iStrike]
        payoff1 = lambda v: (cFunc.evaluate(math.exp(xmk-v))-K)*(1-abs(v)/h)
        payoff1 = np.vectorize(payoff1)
        value = F0smooth[iStrike]
        if isCall:
            a = (X[iStrike]+X[iStrike+1])/2
            #logKinv<0
            value1 = integrate.quad( payoff1, -h,xmk-logKinv)
            value = (value1[0]) /h
        F0smooth[iStrike] = value
    elif smoothing=="KreissF":
        for i in (dIndices):
            xmk = X[i]
            h = hm #(X[i+1]-X[i-1])/2
            sign = 1
            if not isCall:
                sign=-1
            payoff1 = lambda v: max(sign*(cFunc.evaluate(math.exp(xmk-v))-K),0)*(1-abs(v)/h)
            payoff1 = np.vectorize(payoff1)
            value = F0smooth[i]
            value1 = integrate.quad( payoff1, 0,h)
            value0 = integrate.quad( payoff1, -h, 0)
            value = (value0[0]+value1[0]) /h
            #print("new value",value,Xi,iXi)
            F0smooth[i] = value
    elif smoothing=="KreissF4":
        for i in range(M):
            xmk = X[i]
            h = hm #(X[i+1]-X[i-1])/2
            sign = 1
            if not isCall:
                sign=-1
            #             f4 = @(x) (1/36)*(1/2)*...
            # ( +56*x.^3.*sign(x) +(x-3).^3.*(-sign(x-3)) +12*(x-2).^3.*sign(x-2) -39*(x-1).^3.*sign(x-1) -39*(x+1).^3.*sign(x+1) +12*(x+2).^3.*sign(x+2) -(x+3).^3.*sign(x+3));

            payoff1 = lambda v: max(sign*(cFunc.evaluate(math.exp(xmk-v))-K),0)*1.0/72*(56*pow(abs(v/h),3) -pow(abs(v/h-3),3) +12*pow(abs(v/h-2),3) -39*pow(abs(v/h-1),3) -39*pow(abs(v/h+1),3) +12*pow(abs(v/h+2),3) -pow(abs(v/h+3),3))
            payoff1 = np.vectorize(payoff1)
            value = F0smooth[i]
            value1 = integrate.quad( payoff1, -3*h,3*h)
            # value0 = integrate.quad( payoff1, -3*h, 0)
            value = (value1[0]) /h
            #print("new value",value,Xi,iXi)
            F0smooth[i] = value

    #print("F0smooth",F0smooth, len(X), len(V))
    iBarrier = 1
    if not B == 0:
        iBarrier = np.searchsorted(S,B)  #S[i-1]<B<=S[i]
        S=S[iBarrier-1:]
        M=len(S)
        X=X[iBarrier-1:]
        F0smooth = F0smooth[iBarrier-1:]
        J = J[iBarrier-1:]
        Jm = Jm[iBarrier-1:]
        iBarrier=1
    F = np.zeros((L*M))
    for j in range(L):
        F[j*M:(j+1)*M] = F0smooth

    M = len(X)
    L = len(V)
    PX = np.zeros((L*M))
    PV = np.zeros((L*M))
    for j in range(L):
        PX[j*M:(j+1)*M]=X
        PV[j*M:(j+1)*M]=V[j]

    I,dIdX, d2IdX2, dIdV, d2IdV2, d2IdXdV = makeRBFMatrix(PX,PV,cx,cv)
    print("I",I)
    dt = -T/N
    Id = np.identity(L*M)
    #make sure that S[0] = lower bound and S[M]=upper boundary
    #with LS there is no need to solve coeff as preliminary step, as the RHS is known.
    dtPV = 0.5*dt*PV[:,np.newaxis]
    A1 = np.multiply(dtPV,d2IdX2)
    A1 += np.multiply(dt*(r-q)-dtPV,dIdX)
    A1 -=r*dt*I
    A2 = np.multiply(sigma*sigma*dtPV,d2IdV2)
    A2 += np.multiply(dt*kappa*(theta-PV)[:,np.newaxis],dIdV)
    A2 +=np.multiply(dt*rho*sigma*PV[:,np.newaxis],d2IdXdV)
    A = A1+A2
    # print("A1",A1)
    # print("A2",A2)
    # print("A",A)

    #boundary conditions, 0,0, 0,L-1, M-1,0, M-1,L-1.
    # P ( S[0]) = 0 = sum alpha_ij phi((S[0],v)-P_ij) =
    start=time.time()
    for j in range(L):
    #    BC[:,j*M] = I[:,j*M]  #S[0],v
    #    BC[:,M-1+j*M] = I[:,M-1+j*M]  #S[m-1],v
    # or equiv
        A[j*M,:] = 0
        A[M-1+j*M,:] = 0
    # for i in range(1,M-1):
    #     A[i,:] = A1[i,:]
    #     A[i+(L-1)*M,:]=A1[i+(L-1)*M,:]
    #TODO ADD DIRICHLET BC. IS THERE OTHER BC lines? we have here 2*L BC.
    #luI = la.lu_factor(I)
    #coeffPayoff = la.lu_solve(luI,F)
    #payoff = np.dot(I,coeffPayoff)
    #print("payoff",payoff[0:M])
    #raise Exception('toto')

    lu0 = la.lu_factor(I+reg*Id)
    # coeff = la.lu_solve(lu0,F)
    # for spot in spotArray:
    #     logspot = math.log(spot)
    #     phispotv0 = phi(cx,cv,logspot-PX,0-PV)
    #     price = np.dot(phispotv0,coeff)
    #     print(spot,method, price, max(spot-K,0))

    #raise Exception("toto")
    # start=time.time()
    if useDamping:
        a = 0.5
        Li = I+a*A
        lu = la.lu_factor(Li+reg*Id)
        #updatePayoffBoundary(F, S, B, iBarrier, M,L)
        coeff = la.lu_solve(lu,F)
        F = np.dot(I,coeff)
        #updatePayoffBoundary(F, S, B, iBarrier, M,L)
        coeff = la.lu_solve(lu,F)
        F = np.dot(I,coeff)
        N -= 1

    if method =="LS":
        a = 1 - math.sqrt(2)/2
        Li = I+a*A
        lu = la.lu_factor(Li)

        for i in range(N):
            ti=T*(i+1)/(N)
            #updatePayoffBoundary(F, S, B,iBarrier,M,L)
            for j in range(L):
                if isCall:
                    F[0+j*M] = 0 #S[0]*math.exp(-q*T)
                    F[M-1+j*M] =  max(Sc[M-1]*math.exp(-q*ti)-K*math.exp(-r*ti),0)
                else:
                    F[0+j*M] = max(K*math.exp(-r*ti)-Sc[0]*math.exp(-q*ti),0)
                    F[M-1+j*M] = 0
            # for k in range(1,M-1):
            #     F[k] = np.maximum(S[k]*math.exp(-q*ti)-K,0)
            #     F[k+(L-1)*M]=S[k]*math.exp(-q*ti) #A1[i+(L-1)*M,:]

            coeff = la.lu_solve(lu,F) # L*C = F and then I*C
            F1 = np.dot(I,coeff)
            for j in range(L):
                if isCall:
                    F1[0+j*M] = 0 #S[0]*math.exp(-q*T)
                    F1[M-1+j*M] =  max(Sc[M-1]*math.exp(-q*ti)-K*math.exp(-r*ti),0)
                else:
                    F1[0+j*M] =  max(K*math.exp(-r*ti)-Sc[0]*math.exp(-q*ti),0)
                    F1[M-1+j*M] = 0
            # for k in range(1,M-1):
            #     F1[k] = np.maximum(S[k]*math.exp(-q*ti)-K,0) #A1[i,:]
            #     F1[k+(L-1)*M]=S[k]*math.exp(-q*ti) #A1[i+(L-1)*M,:]

            #updatePayoffBoundary(F1, S, B,iBarrier,M,L)
            coeff = la.lu_solve(lu,F1)
            F2 = np.dot(I,coeff)
            for j in range(L):
                if isCall:
                    F2[0+j*M] = 0 #S[0]*math.exp(-q*T)
                    F2[M-1+j*M] = max(Sc[M-1]*math.exp(-q*ti)-K*math.exp(-r*ti),0)
                else:
                    F2[0+j*M] = max(K*math.exp(-r*ti)-Sc[0]*math.exp(-q*ti),0)
                    F2[M-1+j*M] = 0
            # for k in range(1,M-1):
            #     F2[k] = np.maximum(S[k]*math.exp(-q*ti)-K,0)
            #     F2[k+(L-1)*M]=S[k]*math.exp(-q*ti) #A1[i+(L-1)*M,:]

            F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
            #F = np.maximum(F,0)

    else:
        Li = I+A
        lu = la.lu_factor(Li)

        for i in range(N):
            ti=T*(i+1)/(N)
            #updatePayoffBoundary(F,B,iBarrier,M,L)
            for j in range(L):
                if isCall:
                    F[0+j*M] = 0 #S[0]*math.exp(-q*T)
                    F[M-1+j*M] = max(Sc[M-1]*math.exp(-q*ti)-K*math.exp(-r*ti),0)
                else:
                    F[0+j*M] =  max(K*math.exp(-r*ti)-Sc[0]*math.exp(-q*ti),0)
                    F[M-1+j*M] = 0

            coeff = la.lu_solve(lu,F)
            for spot in spotArray:
                logspot = math.log(spot)
                phispotv0 = phi(cx,cv,logspot-PX,v0-PV)
                price = np.dot(phispotv0,coeff)
                print(spot,method,i, price)

            F = np.dot(I,coeff)

    end=time.time()
    coeff = la.lu_solve(lu0,F)
    print("F",F)
    print("coeff",coeff)
    #F[50+4*M]
    #S0=101.52
    # Payoff = F.reshape(L,M)
    # print("Payoff V=0",Payoff[0])
    # jv0 = np.searchsorted(V,v0)
    # print("Payoff V=V0",V[jv0])
    # for (si,pi) in zip(S, Payoff[jv0]):
    #     print(si, pi)
    #
    # # istrike =np.searchsorted(S,K)
    # # print("Payoff S=K",S[istrike])
    # # for (vi,pi) in zip(V, Payoff[:][istrike]):
    # #     print(vi, pi)
    # plt.grid(True)
    # plt.plot(S[:30], Payoff[jv0][:30])
    # #plt.plot(V,Payoff[:][istrike])
    # plt.yscale('symlog',linthreshy=1e-6)
    # plt.show()
    #Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    maxError = 0.0
#    Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    for spot,refPrice in zip(spotArray,priceArray):
        logspot = math.log(spot)
        phispotv0,dphi,d2phi,dphidv,d2phidv2, dphidxdv = phi(cx,cv,logspot-PX,v0-PV)
        price = np.dot(phispotv0,coeff)
        delta = np.dot(dphi,coeff)/spot
        gamma = np.dot(phispotv0,coeff)/(spot*spot)
        error = price -refPrice
        if abs(error) > maxError:
            maxError = abs(error)
        print(spot,method,N,M,L, price, delta,gamma,error)
    if not B==0:
        logspot = math.log(K)
        phispotv0,dphi,d2phi = phi(cx,cv,logspot-PX,v0-PV)
        price = np.dot(phispotv0,coeff)
        print(method,N,M,L,price,end-start)
    else:
        print(method,N,M,L,maxError,end-start)

@jit(nopython=True)
def updatePayoffBoundary(F, S, B, iBarrier, M,L):
    if not B == 0:
        for j in range(L):
            F[j*M:iBarrier +j*M] = 0

@jit(nopython=True)
def updatePayoffExplicit(F, S, B, iBarrier, M,L):
    # Si-B *  Vim + Vi * B-Sim  =0
    if not B == 0:
        for j in range(L):
            F[j*M:iBarrier-1 +j*M] = 0
            F[iBarrier-1 +j*M] = F[iBarrier + j*M] * (S[iBarrier-1]-B)/(S[iBarrier]-B)

def priceBenchopSpace():
    v0=0.0225
    kappa = 2.0
    theta = 0.0225
    sigma = 0.25
    rho = -0.5
    r = 0.03
    q = 0.0
    T = 1.0
    K = 100.0
    B = 0.0
    priceArray = np.array([2.302535842814927, 7.379832496149447, 14.974005277144057])
    spotArray = np.array([90,100,110])
    #priceArray = blackScholes(1,spotArray,K,T,math.sqrt(v0),r,q)

    M = 401 #X
    L = 101 #V
    Ms = [25, 51, 101, 201, 401]
    Ls = [12, 25, 51, 101, 201]
    Ms = [301]
    Ls= [11]
    N = 100 #s = [4,8,16,32,64,128] #timesteps
    for L,M in zip(Ls,Ms):
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, B, N, M, L)


def priceAlbrecherSpace():
    v0=0.04
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    rho = -0.9
    r = 0.025
    q = 0.0
    T = 1.0
    K = 100.0
    B = 0.0
    priceArray = np.array([0.4290429592804125, 0.5727996675731273, 0.7455984677403922, 0.9488855729391782, 1.1836198521834569, 1.4503166421285438, 1.7491038621459454, 2.079782505454696, 2.4418861283930053, 2.834736019523883, 3.257490337101448, 3.709186519701557, 4.188777097589518, 4.6951592762243415, 5.227198998513091, 5.7837501984978665, 6.363669958734282, 6.965830262856437, 7.589126920735202, 8.232486143930792, 8.894869093849636, 9.575277129770623, 10.272748751757314, 10.986365852615036, 11.715254013220457, 12.458577567319875, 13.215544738495424, 13.98540421747423, 14.767442110445812, 15.560982138391632, 16.36538729643898, 17.180051769091545, 18.004405483745735, 18.8379101967189, 19.68005854335592, 20.53036894075123, 21.388390582359417, 22.25369629176841, 23.12588767795124, 24.004578691901752, 24.889416575642677])
    spotArray = np.array([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120])
    #priceArray = blackScholes(1,spotArray,K,T,math.sqrt(v0),r,q)

    M = 401 #X
    L = 101 #V
    Ms = [25, 51, 101, 201, 401]
    Ls = [12, 25, 51, 101, 201]
    Ms = [401]
    Ls= [21]
    N = 32#s = [4,8,16,32,64,128] #timesteps
    for L,M in zip(Ls,Ms):
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, B, N, M, L)


def priceAlbrecherTime():
    v0=0.04
    kappa = 1.5
    theta = 0.04
    sigma = 0.3
    rho = -0.9
    r = 0.025
    q = 0.0
    T = 1.0
    K = 100.0
    B=0 #90.0
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [0.4290429592804125, 0.5727996675731273, 0.7455984677403922, 0.9488855729391782, 1.1836198521834569, 1.4503166421285438, 1.7491038621459454, 2.079782505454696, 2.4418861283930053, 2.834736019523883, 3.257490337101448, 3.709186519701557, 4.188777097589518, 4.6951592762243415, 5.227198998513091, 5.7837501984978665, 6.363669958734282, 6.965830262856437, 7.589126920735202, 8.232486143930792, 8.894869093849636, 9.575277129770623, 10.272748751757314, 10.986365852615036, 11.715254013220457, 12.458577567319875, 13.215544738495424, 13.98540421747423, 14.767442110445812, 15.560982138391632, 16.36538729643898, 17.180051769091545, 18.004405483745735, 18.8379101967189, 19.68005854335592, 20.53036894075123, 21.388390582359417, 22.25369629176841, 23.12588767795124, 24.004578691901752, 24.889416575642677]
    M = 201 #X
    L = 21 #V
    Ns = [512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    for N in Ns:
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K,B, N, M, L)


def priceBloombergSpace():
    kappa = 3.0
    theta = 0.12
    sigma = 0.04
    rho = 0.6 #!FIXME breaks with - sign. : iStrike not in array!?
    r = 0.01
    q = 0.04
    v0=theta
    T=1.0
    K=100.0
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [4.126170747504533, 4.408743197301329, 4.70306357455405, 5.009202471608047, 5.327215893333642, 5.657145552450321, 5.999019203695557, 6.3528510118569015, 6.718641951722364, 7.096380233599666, 7.486041751584794, 7.887590552192177, 8.300979318221902, 8.726149865537172, 9.163033649989693, 9.611552278338717, 10.071618030216948, 10.543134388629074, 11.025996479014745, 11.520091740844437, 12.025300295511904, 12.54149551835306, 13.068544517640353, 13.606308624804461, 14.154643874270963, 14.713401467714998, 15.282428228751144, 15.861567038426507, 16.450657265344518, 17.04953517774978, 17.658034469027065, 18.2759861100527, 18.903219497330056, 19.539562310453945, 20.184840914482272, 20.838880779749626, 21.501506644797566, 22.17254294281439, 22.85181397102651, 23.539144197874872, 24.23435849148654]
    Ms = [25, 51, 101, 201, 401]
    Ls = [5, 5, 5, 11, 11]

    N =64 #s = [4,8,16,32,64,128] #timesteps
    for L,M in zip(Ls,Ms):
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K,0, N, M, L)

def priceBloombergTime():
    kappa = 3.0
    theta = 0.12
    sigma = 0.04
    rho = 0.6
    r = 0.01
    q = 0.04
    v0=theta
    T=1.0
    K=100.0
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [4.126170747504533, 4.408743197301329, 4.70306357455405, 5.009202471608047, 5.327215893333642, 5.657145552450321, 5.999019203695557, 6.3528510118569015, 6.718641951722364, 7.096380233599666, 7.486041751584794, 7.887590552192177, 8.300979318221902, 8.726149865537172, 9.163033649989693, 9.611552278338717, 10.071618030216948, 10.543134388629074, 11.025996479014745, 11.520091740844437, 12.025300295511904, 12.54149551835306, 13.068544517640353, 13.606308624804461, 14.154643874270963, 14.713401467714998, 15.282428228751144, 15.861567038426507, 16.450657265344518, 17.04953517774978, 17.658034469027065, 18.2759861100527, 18.903219497330056, 19.539562310453945, 20.184840914482272, 20.838880779749626, 21.501506644797566, 22.17254294281439, 22.85181397102651, 23.539144197874872, 24.23435849148654]
    M = 201 #X
    L = 101 #V
    Ns = [2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    for N in Ns:
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, N, M, L)


def blackScholes (cp, s, k, t, v, rf, div):
        """ Price an option using the Black-Scholes model.
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        cp: +1/-1 for call/put
        """

        d1 = (np.log(s/k)+(rf-div+0.5*math.pow(v,2))*t)/(v*math.sqrt(t))
        d2 = d1 - v*math.sqrt(t)

        optprice = (cp*s*math.exp(-div*t)*norm.cdf(cp*d1)) - (cp*k*math.exp(-rf*t)*norm.cdf(cp*d2))
        return optprice


class CollocationFunction:
    X = []
    A = []
    B = []
    C = []
    leftSlope = 0.0
    rightSlope = 0.0

    def __init__(self, X, A, B, C,leftSlope,rightSlope):
        self.X = X
        self.A = A
        self.B = B
        self.C = C
        self.leftSlope = leftSlope
        self.rightSlope = rightSlope

    def evaluate(self, z):
        if z <= self.X[0]:
            return self.leftSlope*(z-self.X[0]) + self.A[0]
        elif z >= self.X[-1]:
            return self.rightSlope*(z-self.X[-1])+self.A[-1]
        i = np.searchsorted(self.X,z)  # x[i-1]<z<=x[i]
        if i > 0:
            i -= 1
        h = z-self.X[i]
        return self.A[i] + h*(self.B[i]+h*self.C[i])

    def solve(self, strike):
        if strike < self.A[0]:
            sn = self.leftSlope
            return (strike-self.A[0])/sn + self.X[0]
        elif strike > self.A[-1]:
        	sn = self.rightSlope
        	return (strike-self.A[-1])/sn + self.X[-1]
        i = np.searchsorted(self.A,strike)  # a[i-1]<strike<=a[i]
        # print("index",self.A[i-1],strike,self.A[i],len(self.A))
        if abs(self.A[i]-strike)< 1e-10:
            return self.X[i]
        if abs(self.A[i-1]-strike)< 1e-10:
            return self.X[i-1]
        if i == 0:
            i+=1
        x0 = self.X[i-1]
        c = self.C[i-1]
        b = self.B[i-1]
        a = self.A[i-1]
        d = 0
        cc = a + x0*(-b+x0*(c-d*x0)) - strike
        bb = b + x0*(-2*c+x0*3*d)
        aa = -3*d*x0 + c
        allck = np.roots([aa,bb,cc])
        for ck in allck:
            if abs(ck.imag) < 1e-10 and ck.real >= self.X[i-1]-1e-10 and ck.real <= self.X[i]+1e-10:
                return ck.real
        raise Exception("no roots found in range", allck, strike, aa, bb, cc, i,self.X[i-1],self.X[i])

def priceSX5ETime():
    #Spline 1e-5 pennalty
    A=[0.6287965835693049 ,0.8796805556963849 , 0.9548458991431029 ,0.9978807937190832 ,1.0432949917908245, 1.0951689975427406, 1.1780329537431, 1.2767467611605525]
    B=[0.846962887118158, 0.5006951388813219 ,1.3162296284270554, 0.764281474912235, 1.4312564546785838, 1.0765792448141005, 0.9264392665602718]
    C=[-0.46500629962499923, 4.928351101396242, -6.670948501034147, 8.061184212984527, -4.286695020953507, -0.907309913530479, -1.9936316682418205]
    X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
    leftSlope=0.846962887118158
    rightSlope=0.2666342520834516

    #Spline 1e-3 penalty
    A=np.array([0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061])
    B=np.array([0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941])
    C=np.array([-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938])
    X=np.array([0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636])
    leftSlope=0.8329310535215612
    rightSlope=0.2668764075068484

    # A=X
    # B[:]=1
    # C[:]=0
    # leftSlope = 1
    # rightSlope = 1
    #Absorption 0.001 0
    cFunc = CollocationFunction(X,A,B,C,leftSlope,rightSlope)
    # print("slope left",(cFunc.evaluate(X[0]+1e-7)-cFunc.evaluate(X[0]))/1e-7,leftSlope)
    # print("slope r",(cFunc.evaluate(X[-1]-1e-7)-cFunc.evaluate(X[-1]))/1e-7,rightSlope)
    kappa = 0.35
    theta = 0.321
    sigma = 1.388
    rho = -0.63
    r = 0.0
    q = 0.0
    v0=0.133
    T=0.4986301369863014
    K=1.0
    spotArray = [1.0] #max(s-K) = max(s/K-1)*K
    priceArray = [0.07260310]
    priceArray = [0.07278065]
    # K=0.7
    # spotArray = [1.0]
    # priceArray = [0.30953450-0.3] #P = C- F-K
    # priceArray = [0.00960629]
    # K=1.4
    # spotArray = [1.0]
    # priceArray = [0.00015184+.4]
    # priceArray = [0.40015225]

    M = 30  #X
    L = 30 #V
    B=0
    # Ns = [4096,2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns = [4096,1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
    Ns = [12]
    Ns.reverse()
    for N in Ns:
        priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)

def main():
    #priceBenchopSpace()
    #priceAlbrecherSpace()
    #priceAlbrecherTime()
    # priceBloombergSpace()
    #priceBloombergTime()
    priceSX5ETime()

if __name__ =='__main__':
    main()
