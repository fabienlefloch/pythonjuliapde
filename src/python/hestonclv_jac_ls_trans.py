import numpy as np
import math
import time
from scipy.sparse import csc_matrix, lil_matrix, dia_matrix, identity, linalg as sla
from scipy import linalg as la
from scipy.stats import ncx2
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

def priceCall(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L):
    isCall = False
    method = "LS" # "LS","CN","DO"
    smoothing = "KreissF" #"Kreiss","Averaging","None"
    useDamping = False
    useLinear = False
    useVLinear = False
    useExponentialFitting = True
    upwindingThreshold = 1.0
    epsilon = 1e-4
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*math.exp(-kappa*T)/(sigma*sigma*(1-math.exp(-kappa*T)))
    vmax = ncx2.ppf((1-epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    vmin = ncx2.ppf((epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    vmin = max(1e-4,vmin)
    #print("vmax",vmin,vmax, 10*v0)
    #vmax=10.0*v0
    #vmin = 0
    V = np.arange(L)*(vmax/(L-1))
    W = V
    hl = W[1]-W[0]
    JV=np.ones(L)
    JVm=np.ones(L)

    if not useVLinear:
        vscale = v0
        u = np.linspace(0,1,L) #1e-4,math.sqrt(vmax),L)  #ideally, concentrated around v0: V=sinh((w-w0)/c). w unif
        c1 = math.asinh((vmin-v0)/vscale)
        c2 = math.asinh((vmax-v0)/vscale)
        V = v0 + vscale*np.sinh((c2-c1)*u+c1)
        hl = u[1]-u[0]
        JV = vscale*(c2-c1)* np.cosh((c2-c1)*u+c1)
        JVm = vscale*(c2-c1)* np.cosh((c2-c1)*(u-hl/2)+c1)

    Xspan = 4*math.sqrt(theta*T)
    Xmin = math.log(K) - Xspan + (r-q)*T -0.5*v0*T
    Xmax = math.log(K) + Xspan + (r-q)*T -0.5*v0*T
    X = np.linspace(Xmin,Xmax,M)
    hm = X[1]-X[0]
    #X+=hm/2
    S = np.exp(X)
    J= np.exp(X)
    Jm= np.exp(X-hm/2)

    #S lin
    if useLinear:
        #S=np.linspace(0,K*4,M)
        S=np.linspace(0,math.exp(Xmax),M)
        X=S
        hm = X[1]-X[0]
        # X+=hm/2
        S=X
        J=np.ones(M)
        Jm=np.ones(M)

    cFuncEval = np.vectorize(cFunc.evaluate)
    Sc = np.array([cFunc.evaluate(T, Si) for Si in S])
    if isCall:
        F0 = np.maximum(Sc-K,0)
    else:
        F0 = np.maximum(K-Sc,0)
    F0smooth = np.array(F0,copy=True)
    Kinv = cFunc.solve(K)
    if smoothing == "Averaging":
        iStrike = np.searchsorted(S,Kinv)  # S[i-1]<K<=S[i]
        if Kinv < (S[iStrike]+S[iStrike-1])/2:
            iStrike -= 1
        payoff1 = lambda v: cFunc.evaluate(T,v)-K
        payoff1 = np.vectorize(payoff1)
        value = 0
        if isCall:
            a = (S[iStrike]+S[iStrike+1])/2
            value = integrate.quad( payoff1, Kinv, a)
        else:
            a = (S[iStrike]+S[iStrike-1])/2   # int a,lnK K-eX dX = K(a-lnK)+ea-K
            value = integrate.quad( payoff1, Kinv, a)
        h = (S[iStrike+1]-S[iStrike-1])/2
        F0smooth[iStrike] = value[0]/h

    elif smoothing == "Kreiss":
        iStrike = np.searchsorted(S,Kinv)  # S[i-1]<K<=S[i]
        xmk = S[iStrike]
        h = (S[iStrike+1]-S[iStrike-1])/2
        payoff1 = lambda v: (cFunc.evaluate(T,xmk-v)-K)*(1-abs(v)/h)
        payoff1 = np.vectorize(payoff1)
        value = F0smooth[iStrike]
        if isCall:
            value1 = integrate.quad( payoff1, 0,xmk-Kinv)
            value0 = integrate.quad( payoff1, -h, 0)
            value = (value0[0]+value1[0]) /h

        F0smooth[iStrike] = value
        iStrike -= 1
        xmk = S[iStrike]
        payoff1 = lambda v: (cFunc.evaluate(T,xmk-v)-K)*(1-abs(v)/h)
        payoff1 = np.vectorize(payoff1)
        value = F0smooth[iStrike]
        if isCall:
            value1 = integrate.quad( payoff1, -h,xmk-Kinv)
            value = (value1[0]) /h
        F0smooth[iStrike] = value
    elif smoothing=="KreissF":
        for i in range(M):
            xmk = S[i]
            sign = 1
            if not isCall:
                sign = -1
            h = hm #(X[i+1]-X[i-1])/2
            payoff1 = lambda v: max(sign*(cFunc.evaluate(T,xmk-v)-K),0)*(1-abs(v)/h)
            payoff1 = np.vectorize(payoff1)
            value = F0smooth[i]
            value1 = integrate.quad( payoff1, 0,h)
            value0 = integrate.quad( payoff1, -h, 0)
            value = (value0[0]+value1[0]) /h
            #print("new value",value,Xi,iXi)
            F0smooth[i] = value

    elif smoothing=="KreissF4":
        for i in range(M):
            xmk = S[i]
            sign = 1
            if not isCall:
                sign = -1
            h = hm #(X[i+1]-X[i-1])/2
            payoff1 = lambda v: max(sign*(cFunc.evaluate(T,xmk-v)-K),0)*1.0/72*(56*pow(abs(v/h),3) -pow(abs(v/h-3),3) +12*pow(abs(v/h-2),3) -39*pow(abs(v/h-1),3) -39*pow(abs(v/h+1),3) +12*pow(abs(v/h+2),3) -pow(abs(v/h+3),3))
            payoff1 = np.vectorize(payoff1)
            value = F0smooth[i]
            value1 = integrate.quad( payoff1, 0,3*h)
            value0 = integrate.quad( payoff1, -3*h, 0)
            value = (value0[0]+value1[0]) /h
            #print("new value",value,Xi,iXi)
            F0smooth[i] = value
    #print("F0smooth",F0smooth)
    iBarrier = 1
    if not B == 0:
        iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
    F = []
    for j in range(L):
        F =  np.append(F,F0smooth)

    dt = -T/N
    A01 = lil_matrix((L*M,L*M))
    A02 = lil_matrix((L*M,L*M))
    A1 = lil_matrix((L*M,L*M))
    A2 = lil_matrix((L*M,L*M))
    #boundary conditions, 0,0, 0,L-1, M-1,0, M-1,L-1.
    i=0
    j=0
    #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i+1)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+1+(j+1)*M]+=rij
    #A[i+j*M,i+1+(j)*M]+=-rij
    #A[i+j*M,i+(j+1)*M]+=-rij
    #A[i+j*M,i+(j)*M]+=rij
    i=0
    j=L-1
    #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i+1)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += rij
    #A[i+j*M,i+1+(j-1)*M]+=-rij
    #A[i+j*M,i+1+(j)*M]+=rij
    #A[i+j*M,i+(j-1)*M]+=rij
    #A[i+j*M,i+(j)*M]+=-rij

    i=M-1
    j=L-1
    rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i-1+(j-1)*M]+=rij
    #A[i+j*M,i-1+(j)*M]+=-rij
    #A[i+j*M,i+(j-1)*M]+=-rij
    #A[i+j*M,i+(j)*M]+=rij
    i=M-1
    j=0
    rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j+1)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i-1+(j+1)*M]+=-rij
    #A[i+j*M,i-1+(j)*M]+=rij
    #A[i+j*M,i+(j+1)*M]+=rij
    #A[i+j*M,i+(j)*M]+=-rij

    for i in range(1,M-1):
        j=0
        svi = S[i]*S[i]*V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
        drifti = (r-q)*S[i]
        if useExponentialFitting:
            if  svi > 0 and abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                #svi = svi +0.5*abs(drifti)*hm

        svi = svi/(2*hm*hm)
        A1[i+j*M,(i+1)+j*M] += dt*(svi/Jm[i+1]+drifti/(2*J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-svi*(1.0/Jm[i+1]+1.0/Jm[i])-r*0.5)
        A1[i+j*M,(i-1)+j*M] += dt*(svi/Jm[i]-drifti/(2*J[i]*hm))
        A2[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
        #A[i+j*M,i+1+(j+1)*M]+=rij
        #A[i+j*M,i+1+(j)*M]+=-rij
        #A[i+j*M,i+(j+1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=rij
        j=L-1
        svi = S[i]*S[i]*V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
        drifti = (r-q)*S[i]
        if useExponentialFitting:
            if svi > 0 and abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                #svi = svi +0.5*abs(drifti)*hm

        svi = svi/(2*hm*hm)
        rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
        A1[i+j*M,(i-1)+j*M] += dt*(svi/Jm[i]-drifti/(2*J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-svi*(1.0/Jm[i+1]+1.0/Jm[i])-r*0.5)
        A1[i+j*M,(i+1)+j*M] += dt*(svi/Jm[i+1]+drifti/(2*J[i]*hm))
        A2[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i-1+(j-1)*M]+=rij
        #A[i+j*M,i-1+(j)*M]+=-rij
        #A[i+j*M,i+(j-1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=rij

    for j in range(1,L-1):
        #boundary conditions i=0,M-1.
        i=0
        A1[i+j*M,(i+1)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)

        i=M-1
        rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
        A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)*S[i]/(J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*((r-q)*S[i]/(J[i]*hm))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i-1+(j-1)*M]+=rij
        #A[i+j*M,i-1+(j)*M]+=-rij
        #A[i+j*M,i+(j-1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=+rij
        for i in range(1,M-1):
            svi = S[i]*S[i]*V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
            svj = sigma*sigma*V[j]/(JV[j])
            drifti = (r-q)*S[i]
            driftj = kappa*(theta-V[j])
            if useExponentialFitting:
                if abs(drifti*hm/svi) > upwindingThreshold:
                    svi = drifti*hm/math.tanh(drifti*hm/svi)
                    # svi = svi +0.5*abs(drifti)*hm
                if driftj != 0 and abs(driftj*hl/svj) > 1.0:
                    # svj = svj +0.5*abs(driftj)*hl
                    svj = driftj*hl/math.tanh(driftj*hl/svj)

            rij = dt*0.25*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
            A1[i+j*M,(i+1)+j*M] += dt*(0.5*svi/(hm*hm*Jm[i+1])+drifti/(2*J[i]*hm))
            A1[i+j*M,i+j*M] += dt*(-svi*0.5/(hm*hm)*(1.0/Jm[i+1]+1.0/Jm[i]) -r*0.5)
            A1[i+j*M,(i-1)+j*M] += dt*(0.5*svi/(hm*hm*Jm[i])-drifti/(2*J[i]*hm))
            A2[i+j*M,i+(j+1)*M] += dt*(0.5*svj/(hl*hl*JVm[j+1])+driftj/(2*JV[j]*hl))
            A2[i+j*M,i+j*M] += dt*(-r*0.5-svj*0.5/(hl*hl)*(1.0/JVm[j+1]+1.0/JVm[j]))
            A2[i+j*M,i+(j-1)*M] += dt*(svj*0.5/(JVm[j]*hl*hl)-driftj/(2*JV[j]*hl))
            A01[i+j*M,i+1+(j+1)*M]+= rij
            A02[i+j*M,i+1+(j-1)*M]+=-rij
            A02[i+j*M,i-1+(j+1)*M]+=-rij
            A01[i+j*M,i-1+(j-1)*M]+=rij
            A01[i+j*M,i+(j)*M]+=-2*rij
            A02[i+j*M,i+(j)*M]+=2*rij


    A0 = (A01+A02).tolil()
    #print((A0+A1+A2).shape)
    # print((A0+A1+A2)[:,1000].getnnz())
    #plt.spy(A0+A1+A2,markersize=1)
    #plt.show()
    #ax  = plot_coo_matrix(A0+A1+A2)
    #ax.figure.show(block=True)
    #plt.show(ax.figure)
    #raise Error
    I = identity(M*L,format="csc")
    A0Full = A0.copy()
    A1Full = A1.copy()
    A2Full = A2.copy()

    BC = lil_matrix((L*M,L*M))

    #boundary conditions j=0,L-1.

    start=time.time()
    if useDamping:
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A + BC
        lu = sla.splu(Li)
        updatePayoffBoundary(F, Sc, B, iBarrier, M,L)
        F = lu.solve(F)
        updatePayoffBoundary(F, Sc, B, iBarrier, M,L)
        F = lu.solve(F)
        N -= 1

    if method == "CS":
        a = 0.5
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)
        for i in range(N):
            #updatePayoffExplicit(F, S, B, iBarrier, M,L)
            Y0 = (I-A0-A1-A2)*F #explicit
            #updatePayoffExplicit(Y0, S, B, iBarrier, M,L)
            Y0r = Y0+a*A1*F
            updatePayoffBoundary(Y0r, Sc, B, iBarrier, M,L)
            Y1 = lu1.solve(Y0r)
            Y1r = Y1+a*A2*F
            updatePayoffBoundary(Y1r, Sc, B, iBarrier, M,L)
            Y2 = lu2.solve(Y1r)
            Y0t = Y0 - 0.5*(A0*Y2-A0*F)
            Y0r = Y0t+a*A1*F
            updatePayoffBoundary(Y0r, Sc, B, iBarrier, M,L)
            Y1t = lu1.solve(Y0r)
            Y1r = Y1t+a*A2*F
            updatePayoffBoundary(Y1r, Sc, B, iBarrier, M,L)
            Y2t = lu2.solve(Y1r)
            F = Y2t

    elif method == "HW":
        a = 0.5+math.sqrt(3)/6
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)
        for i in range(N):
            #updatePayoffExplicit(F, S, B, iBarrier, M,L)
            Y0 = (I-A0-A1-A2)*F #explicit
            #updatePayoffExplicit(Y0, S, B, iBarrier, M,L)
            Y0 = Y0+a*A1*F
            updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
            Y1 = lu1.solve(Y0)
            Y1 = Y1+a*A2*F
            updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
            Y2 = lu2.solve(Y1)
            #updatePayoffExplicit(Y2, S, B, iBarrier, M,L)
            Y0 = F-0.5*(A0+A1+A2)*(F+Y2)
            Y0 = Y0+a*A1*Y2
            updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
            Y1 = lu1.solve(Y0)
            Y1 = Y1+a*A2*Y2
            updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
            Y2 = lu2.solve(Y1)
            F = Y2

    elif method == "DO":
        a = 0.5
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)
        for i in range(N):
            updatePayoffExplicit(F, Sc, B, iBarrier, M,L)
            Y0 = (I-A0-A1-A2+BC)*F #explicit
            updatePayoffExplicit(Y0, Sc, B, iBarrier, M,L)
            Y0 = Y0+a*A1*F
            updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
            Y1 = lu1.solve(Y0)
            Y1 = Y1+a*A2*F
            updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
            Y2 = lu2.solve(Y1)
            F = Y2
    elif method == "LODLS":
        a = 1 - math.sqrt(2)/2
        Li = I+a*A1
        lu = sla.splu(Li)
        lu2 = sla.splu(I+a*(A0+A2))

        for i in range(N):
            F1 = lu.solve(F)
            F1t = lu2.solve(F1)
            F2 = lu.solve(F1t)
            F2t = lu2.solve(F2)
            F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F1t
            #F = np.maximum(F,0)

    elif method == "CN":
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A+BC
        Le = I-(1-a)*A
        lu = sla.splu(Li)
        for i in range(N):
            #updatePayoffExplicit(F, S, B, iBarrier, M, L)
            F1 = Le*F
            updatePayoffBoundary(F1, Sc, B, iBarrier, M,L)
            F = lu.solve(F1)
    elif method =="LS":
        a = 1 - math.sqrt(2)/2
        if not B == 0:
            for i in range(N):
                ti = T*(N-i-1)/N
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
                # print("ti",ti,"iB",iBarrier, M,Sc,B)
                A0 = A0Full.copy()
                A1 = A1Full.copy()
                A2 = A2Full.copy()
                BC = lil_matrix((L*M,L*M))
                updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
                A = A0+A1+A2
                Li = I+a*A+BC #FIXME compute A from 0, then update rows according to BC as iBarrier moves!
                lu = sla.splu(Li)
                updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
                F1 = lu.solve(F)
                updatePayoffBoundary(F1, Sc, B,iBarrier,M,L)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
                #F = np.maximum(F,0)
        else:
            A0 = A0Full.copy()
            A1 = A1Full.copy()
            A2 = A2Full.copy()
            BC = lil_matrix((L*M,L*M))
            updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
            A = A0+A1+A2
            Li = I+a*A+BC #FIXME compute A from 0, then update rows according to BC as iBarrier moves!
            lu = sla.splu(Li)
            for i in range(N):
                updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
                F1 = lu.solve(F)
                updatePayoffBoundary(F1, Sc, B,iBarrier,M,L)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
    elif method == "O4":
        A = A0+A1+A2
        # a1 = 1.0/(6 - 2*math.sqrt(6))
        # a2 = 1.0/(2*(3+math.sqrt(6)))
        # lu1 = sla.splu(I + a1*A+BC)
        # lu2 = sla.splu(I + a2*A+BC)
        Asq = A*A
        Li0 = I+A+0.5*Asq+1.0/6*A*Asq
        lu0 = sla.splu(Li0+BC)
        lu = sla.splu(I+0.5*A+1.0/12*Asq+BC)
        #F0 = F - A*F + 0.5*A*A*F - 1.0/6* A*A*A*F
        #F1 = F0 -  A*F0 + 0.5*A*A*F0 - 1.0/6* A*A*A*F0# A*F0 + 0.5*A*(I-A/3)*(A*F0)

        updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
        F0 = lu0.solve(F)
        updatePayoffBoundary(F0, Sc, B,iBarrier,M,L)
        F1 = lu0.solve(F0)
        F = F1
        for i in range(N-2):
            Fr= F-0.5*A*(F - 1.0/6*A*F)
            updatePayoffBoundary(Fr, Sc, B,iBarrier,M,L)
            # F1 = lu2.solve(Fr)
            # updatePayoffBoundary(F1, S, B,iBarrier,M,L)
            F = lu.solve(Fr)

    else:
        if not B == 0:
            for i in range(N):
                ti = T*(N-i-1)/N
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
                # print("ti",ti,"iB",iBarrier, M,Sc,B)
                A0 = A0Full.copy()
                A1 = A1Full.copy()
                A2 = A2Full.copy()
                BC = lil_matrix((L*M,L*M))
                updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
                A = A0+A1+A2
                Li = I+A+BC
                lu = sla.splu(Li)
                updatePayoffBoundary(F,Sc,B,iBarrier,M,L)
                F = lu.solve(F)
        else:
            A0 = A0Full.copy()
            A1 = A1Full.copy()
            A2 = A2Full.copy()
            BC = lil_matrix((L*M,L*M))
            updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
            A = A0+A1+A2
            Li = I+A+BC
            lu = sla.splu(Li)
            for i in range(N):
                updatePayoffBoundary(F,Sc,B,iBarrier,M,L)
                F = lu.solve(F)


    end=time.time()

    #F[50+4*M]
    #S0=101.52
    Payoff = F.reshape(L,M)
    #print("Payoff V=0",Payoff[0])
    jv0 = np.searchsorted(V,v0)
    #print("Payoff V=V0",V[jv0])
    #for (si,pi) in zip(S, Payoff[jv0]):
    #    print(si, pi)
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
    Payoffi = interpolate.RectBivariateSpline(V,S,Payoff,kx=3,ky=3,s=0)
    maxError = 0.0
#    Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    #print("spot method n m l price delta gamma error")
    for spot,refPrice in zip(spotArray,priceArray):
        price = Payoffi(v0,spot)[0][0]
        delta = Payoffi(v0,spot,dy=1)[0][0]
        gamma = Payoffi(v0,spot,dy=2)[0][0]
        error = price -refPrice
        if abs(error) > maxError:
            maxError = abs(error)
        # print(spot,method,N,M,L, price, delta,gamma,error)
    if not B==0:
        print(method,N,M,L,Payoffi(v0,K)[0][0],end-start)
    else:
        print(method,N,M,L,maxError,end-start)

def updateSystemBoundary(L,M,iBarrier,Sc,B,A0,A1,A2,BC):
    # print(type(A0),type(A1),type(A2))
    for i in range(1,iBarrier):
        j=0
        A2[i+j*M,i+(j+1)*M] =0
        A2[i+j*M,i+j*M] = 0
        j=L-1
        A2[i+j*M,i+(j-1)*M] =0
        A2[i+j*M,i+j*M] = 0
        for j in range(1,L-1):
            A2[i+j*M,i+(j+1)*M] =0
            A2[i+j*M,i+j*M] = 0
            A2[i+j*M,i+(j-1)*M] =0
            A0[i+j*M,i+1+(j+1)*M]= 0
            A0[i+j*M,i+1+(j-1)*M]=0
            A0[i+j*M,i-1+(j+1)*M]=0
            A0[i+j*M,i-1+(j-1)*M]=0
            A0[i+j*M,i+(j)*M]=0
            A0[i+j*M,i+(j)*M]=0
        for j in range(0,L):
            A1[i+j*M,(i+1)+j*M] = 0
            A1[i+j*M,i+j*M] = 0
            A1[i+j*M,(i-1)+j*M] = 0
    updateBoundaryCondition(L,M,iBarrier,Sc,B,BC)

def updateBoundaryCondition(L,M,iBarrier, Sc, B, BC):
    for j in range(L):
        BC[iBarrier-1+j*M,iBarrier-1+j*M]=(Sc[iBarrier]-B)/(Sc[iBarrier]-Sc[iBarrier-1])-1
        BC[iBarrier-1+j*M,iBarrier+j*M]=(B-Sc[iBarrier-1])/(Sc[iBarrier]-Sc[iBarrier-1])

def priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L):
    isCall = False #True
    method = "LS" # "LS","CN","DO"
    smoothing = "KreissF" #"Kreiss","Averaging","None"
    useDamping = False
    useLinear = False
    useVLinear = False #False
    useExponentialFitting = True
    upwindingThreshold = 1.0
    epsilon = 1e-3
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*math.exp(-kappa*T)/(sigma*sigma*(1-math.exp(-kappa*T)))
    vmax = ncx2.ppf((1-epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    vmin = ncx2.ppf((epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    #print("vmax",vmin,vmax, 10*v0)
    vmin = max(1e-4,vmin)
    V = np.arange(L)*(vmax/(L-1))
    W = V
    hl = W[1]-W[0]
    JV=np.ones(L)
    JVm=np.ones(L)

    if not useVLinear:
        vscale = v0*2
        u = np.linspace(0,1,L) #1e-4,math.sqrt(vmax),L)  #ideally, concentrated around v0: V=sinh((w-w0)/c). w unif
        c1 = math.asinh((vmin-v0)/vscale)
        c2 = math.asinh((vmax-v0)/vscale)
        V = v0 + vscale*np.sinh((c2-c1)*u+c1)
        hl = u[1]-u[0]
        JV = vscale*(c2-c1)* np.cosh((c2-c1)*u+c1)
        JVm = vscale*(c2-c1)* np.cosh((c2-c1)*(u-hl/2)+c1)


    Xspan = 4*math.sqrt(theta*T)
    logK = math.log(K) #f(e^zi) = K
    Kinv = cFunc.solve(K)
    logKinv = math.log(Kinv)
   ecol Xmin = logK - Xspan + (r-q)*T - 0.5*v0*T
    Xmax = logK + Xspan + (r-q)*T - 0.5*v0*T
    # Xmin = math.log(cFunc.solve(math.exp(Xmin)))
    # Xmax = math.log(cFunc.solve(math.exp(Xmax)))
    # print(Xmin,Xmax,logKinv)
    X = np.linspace(Xmin,Xmax,M)
    hm = X[1]-X[0]
    S = np.exp(X)
    J=np.ones(M)
    Jm=np.ones(M)

    #S lin
    if useLinear:
        #S=np.linspace(0,K*4,M)
        #X-=hm/2
        S=np.exp(X)
        J=np.ones(M)
        Jm=np.ones(M)

    #V
    cFuncEval = np.vectorize(cFunc.evaluate)
    Sc = np.array([cFunc.evaluate(T,Si) for Si in S])
    # plt.grid(True)
    # # plt.plot(S,Sc)
    # plt.plot(S, [(cFunc.evaluate(Si+1e-5)-cFunc.evaluate(Si))/1e-5 for Si in S])
    # plt.show()

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
        payoff1 = lambda v: cFunc.evaluate(T,math.exp(v))-K
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
        payoff1 = lambda v: (cFunc.evaluate(T,math.exp(xmk-v))-K)*(1-abs(v)/h)
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
        payoff1 = lambda v: (cFunc.evaluate(T,math.exp(xmk-v))-K)*(1-abs(v)/h)
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
            payoff1 = lambda v: max(sign*(cFunc.evaluate(T,math.exp(xmk-v))-K),0)*(1-abs(v)/h)
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

            payoff1 = lambda v: max(sign*(cFunc.evaluate(T,math.exp(xmk-v))-K),0)*1.0/72*(56*pow(abs(v/h),3) -pow(abs(v/h-3),3) +12*pow(abs(v/h-2),3) -39*pow(abs(v/h-1),3) -39*pow(abs(v/h+1),3) +12*pow(abs(v/h+2),3) -pow(abs(v/h+3),3))
            payoff1 = np.vectorize(payoff1)
            value = F0smooth[i]
            value1 = integrate.quad( payoff1, -3*h,3*h)
            # value0 = integrate.quad( payoff1, -3*h, 0)
            value = (value1[0]) /h
            #print("new value",value,Xi,iXi)
            F0smooth[i] = value

    #print("F0smooth",F0smooth)
    iBarrier = 1
    if not B == 0:
        iBarrier = np.searchsorted(S,B)  #S[i-1]<B<=S[i]
    F = []
    for j in range(L):
        F =  np.append(F,F0smooth)

    dt = -T/N
    A01 = lil_matrix((L*M,L*M))
    A02 = lil_matrix((L*M,L*M))
    A1 = lil_matrix((L*M,L*M))
    A2 = lil_matrix((L*M,L*M))
    BC = lil_matrix((L*M,L*M))
    #boundary conditions, 0,0, 0,L-1, M-1,0, M-1,L-1.
    i=0
    j=0
    #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i+1)+j*M] += dt*((r-q)/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*(-(r-q)/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+1+(j+1)*M]+=rij
    #A[i+j*M,i+1+(j)*M]+=-rij
    #A[i+j*M,i+(j+1)*M]+=-rij
    #A[i+j*M,i+(j)*M]+=rij
    i=0
    j=L-1
    #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
    A1[i+j*M,(i+1)+j*M] += dt*((r-q)/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*(-(r-q)/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += rij
    #A[i+j*M,i+1+(j-1)*M]+=-rij
    #A[i+j*M,i+1+(j)*M]+=rij
    #A[i+j*M,i+(j-1)*M]+=rij
    #A[i+j*M,i+(j)*M]+=-rij

    i=M-1
    j=L-1
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    i=M-1
    j=0
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(J[i]*hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)/(J[i]*hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j+1)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i-1+(j+1)*M]+=-rij
    #A[i+j*M,i-1+(j)*M]+=rij
    #A[i+j*M,i+(j+1)*M]+=rij
    #A[i+j*M,i+(j)*M]+=-rij

    #boundary conditions j=0,L-1.
    for i in range(1,M-1):
        j=0
        svi = V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
        drifti = (r-q-0.5*V[j])
        if useExponentialFitting:
            if  svi > 0 and abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                #svi = svi +0.5*abs(drifti)*hm

        svi = svi/(2*hm*hm)
        A1[i+j*M,(i+1)+j*M] += dt*(svi/Jm[i+1]+drifti/(2*J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-svi*(1.0/Jm[i+1]+1.0/Jm[i])-r*0.5)
        A1[i+j*M,(i-1)+j*M] += dt*(svi/Jm[i]-drifti/(2*J[i]*hm))
        A2[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
        #A[i+j*M,i+1+(j+1)*M]+=rij
        #A[i+j*M,i+1+(j)*M]+=-rij
        #A[i+j*M,i+(j+1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=rij
        j=L-1
        svi = V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
        drifti = (r-q-0.5*V[j])
        if useExponentialFitting:
            if svi > 0 and abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                #svi = svi +0.5*abs(drifti)*hm

        svi = svi/(2*hm*hm)
        rij = dt*rho*sigma*V[j]/(JV[j]*J[i]*hl*hm)
        A1[i+j*M,(i-1)+j*M] += dt*(svi/Jm[i]-drifti/(2*J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-svi*(1.0/Jm[i+1]+1.0/Jm[i])-r*0.5)
        A1[i+j*M,(i+1)+j*M] += dt*(svi/Jm[i+1]+drifti/(2*J[i]*hm))
        A2[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i-1+(j-1)*M]+=rij
        #A[i+j*M,i-1+(j)*M]+=-rij
        #A[i+j*M,i+(j-1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=rij

    for j in range(1,L-1):
        #boundary conditions i=0,M-1.
        i=0
        #rij = dt*rho*sigma*V[j]*S[i]/(JV[j]*J[i]*hl*hm)
        A1[i+j*M,(i+1)+j*M] += dt*((r-q)/(J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*(-(r-q)/(J[i]*hm))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        #A2[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        #A2[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        #A0[i+j*M,i+1+(j+1)*M]+=rij
        #A0[i+j*M,i+1+(j)*M]+=-rij
        #A0[i+j*M,i+(j+1)*M]+=-rij
        #A0[i+j*M,i+(j)*M]+=rij

        i=M-1
        A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(J[i]*hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*((r-q)/(J[i]*hm))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i-1+(j-1)*M]+=rij
        #A[i+j*M,i-1+(j)*M]+=-rij
        #A[i+j*M,i+(j-1)*M]+=-rij
        #A[i+j*M,i+(j)*M]+=+rij
        for i in range(1,M-1):
            svi = V[j]/(J[i]) #J[j] = Jacobian(X_j), Jm[j]=Jacobian(Xj-hm/2), S[j]=S(Xj)
            svj = sigma*sigma*V[j]/(JV[j])
            drifti = (r-q-0.5*V[j])
            driftj = kappa*(theta-V[j])
            if useExponentialFitting:
                if abs(drifti*hm/svi) > upwindingThreshold:
                    svi = drifti*hm/math.tanh(drifti*hm/svi)
                    # svi = svi +0.5*abs(drifti)*hm
                if driftj != 0 and abs(driftj*hl/svj) > 1.0:
                    # svj = svj +0.5*abs(driftj)*hl
                    svj = driftj*hl/math.tanh(driftj*hl/svj)

            rij = dt*0.25*rho*sigma*V[j]/(JV[j]*J[i]*hl*hm)
            A1[i+j*M,(i+1)+j*M] += dt*(0.5*svi/(hm*hm*Jm[i+1])+drifti/(2*J[i]*hm))
            A1[i+j*M,i+j*M] += dt*(-svi*0.5/(hm*hm)*(1.0/Jm[i+1]+1.0/Jm[i]) -r*0.5)
            A1[i+j*M,(i-1)+j*M] += dt*(0.5*svi/(hm*hm*Jm[i])-drifti/(2*J[i]*hm))
            A2[i+j*M,i+(j+1)*M] += dt*(0.5*svj/(hl*hl*JVm[j+1])+driftj/(2*JV[j]*hl))
            A2[i+j*M,i+j*M] += dt*(-r*0.5-svj*0.5/(hl*hl)*(1.0/JVm[j+1]+1.0/JVm[j]))
            A2[i+j*M,i+(j-1)*M] += dt*(svj*0.5/(JVm[j]*hl*hl)-driftj/(2*JV[j]*hl))
            A01[i+j*M,i+1+(j+1)*M]+= rij
            A02[i+j*M,i+1+(j-1)*M]+=-rij
            A02[i+j*M,i-1+(j+1)*M]+=-rij
            A01[i+j*M,i-1+(j-1)*M]+=rij
            A01[i+j*M,i+(j)*M]+=-2*rij
            A02[i+j*M,i+(j)*M]+=2*rij

    A0 = (A01+A02).tolil()
    A0Full=A0.copy()
    A1Full=A1.copy()
    A2Full=A2.copy()
    A1tri, A2tri, indices, indicesInv = createTridiagonalIndices(M,L)
    A1tri[1,:] = A1.diagonal(k=0)
    A1tri[-1,:-1] = A1.diagonal(k=-1)
    A1tri[0,1:] = A1.diagonal(k=1)
    A2tri[1,:] = A2.diagonal(k=0)[indicesInv]
    A2i = A2[:,indicesInv]
    A2i = A2i[indicesInv,:]
    #print("size",A2i.shape)
    A2tri[-1,:-1] = A2i.diagonal(k=-1)
    A2tri[0,1:] = A2i.diagonal(k=1)
    #print((A0+A1+A2).shape)
    # print((A0+A1+A2)[:,1000].getnnz())
    #plt.spy(A0+A1+A2,markersize=1)
    #plt.show()
    #raise Error
    I = identity(M*L,format="csc")
    start=time.time()
    if useDamping:
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A + BC
        lu = sla.splu(Li)
        updatePayoffBoundary(F, Sc, B, iBarrier, M,L)
        F = lu.solve(F)
        updatePayoffBoundary(F, Sc, B, iBarrier, M,L)
        F = lu.solve(F)
        N -= 1

    if method == "CS":
        a = 0.5
        if B == 0:
            craigSneydBanded(N, F, I, A0, A1, A2,A1tri,A2tri, indices, indicesInv)
        else:
            lu1 = sla.splu(I+a*A1+BC)
            lu2 = sla.splu(I+a*A2+BC)
            for i in range(N):
                #updatePayoffExplicit(F, S, B, iBarrier, M,L)
                Y0 = (I-A0-A1-A2)*F #explicit
                #updatePayoffExplicit(Y0, S, B, iBarrier, M,L)
                Y0r = Y0+a*A1*F
                updatePayoffBoundary(Y0r, Sc, B, iBarrier, M,L)
                Y1 = lu1.solve(Y0r)
                Y1r = Y1+a*A2*F
                updatePayoffBoundary(Y1r, Sc, B, iBarrier, M,L)
                Y2 = lu2.solve(Y1r)
                Y0t = Y0 - 0.5*(A0*Y2-A0*F)
                Y0r = Y0t+a*A1*F
                updatePayoffBoundary(Y0r, Sc, B, iBarrier, M,L)
                Y1t = lu1.solve(Y0r)
                Y1r = Y1t+a*A2*F
                updatePayoffBoundary(Y1r, Sc, B, iBarrier, M,L)
                Y2t = lu2.solve(Y1r)
                F = Y2t

    elif method == "HV":
        a = 0.5+math.sqrt(3)/6
        if B == 0:
            F = hundsdorferVerwerBanded(N, F, I, A0, A1, A2,A1tri,A2tri, indices, indicesInv)
        else:
            lu1 = sla.splu(I+a*A1+BC)
            lu2 = sla.splu(I+a*A2+BC)
            for i in range(N):
                #updatePayoffExplicit(F, S, B, iBarrier, M,L)
                Y0 = (I-A0-A1-A2)*F #explicit
                #updatePayoffExplicit(Y0, S, B, iBarrier, M,L)
                Y0 = Y0+a*A1*F
                updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
                Y1 = lu1.solve(Y0)
                Y1 = Y1+a*A2*F
                updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
                Y2 = lu2.solve(Y1)
                #updatePayoffExplicit(Y2, S, B, iBarrier, M,L)
                Y0 = F-0.5*(A0+A1+A2)*(F+Y2)
                Y0 = Y0+a*A1*Y2
                updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
                Y1 = lu1.solve(Y0)
                Y1 = Y1+a*A2*Y2
                updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
                Y2 = lu2.solve(Y1)
                F = Y2

    elif method == "DO":
        a = 0.5
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)
        for i in range(N):
            updatePayoffExplicit(F, Sc, B, iBarrier, M,L)
            Y0 = (I-A0-A1-A2+BC)*F #explicit
            updatePayoffExplicit(Y0, Sc, B, iBarrier, M,L)
            Y0 = Y0+a*A1*F
            updatePayoffBoundary(Y0, Sc, B, iBarrier, M,L)
            Y1 = lu1.solve(Y0)
            Y1 = Y1+a*A2*F
            updatePayoffBoundary(Y1, Sc, B, iBarrier, M,L)
            Y2 = lu2.solve(Y1)
            F = Y2
    elif method == "LODLS":
        a = 1 - math.sqrt(2)/2
        Li = I+a*A1
        lu = sla.splu(Li)
        lu2 = sla.splu(I+a*(A0+A2))

        for i in range(N):
            F1 = lu.solve(F)
            F1t = lu2.solve(F1)
            F2 = lu.solve(F1t)
            F2t = lu2.solve(F2)
            F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F1t
            #F = np.maximum(F,0)

    elif method == "CN":
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A+BC
        Le = I-(1-a)*A
        lu = sla.splu(Li)
        for i in range(N):
            #updatePayoffExplicit(F, S, B, iBarrier, M, L)
            F1 = Le*F
            updatePayoffBoundary(F1, Sc, B, iBarrier, M,L)
            F = lu.solve(F1)
    elif method =="LS":
        a = 1 - math.sqrt(2)/2
        if B==0:
            A = A0+A1+A2
            Li = I+a*A+BC
            lu = sla.splu(Li)
            for i in range(N):
                updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
                F1 = lu.solve(F)
                updatePayoffBoundary(F1, Sc, B,iBarrier,M,L)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
                #F = np.maximum(F,0)
        else:
            for i in range(N):
                ti = T*(N-i)/N
                ti += a*dt
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
                # print("ti",ti,"iB",iBarrier, M,Sc,B)
                A0 = A0Full.copy()
                A1 = A1Full.copy()
                A2 = A2Full.copy()
                BC = lil_matrix((L*M,L*M))
                updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
                A = A0+A1+A2
                Li = I+a*A+BC #FIXME compute A from 0, then update rows according to BC as iBarrier moves!
                lu = sla.splu(Li)
                updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
                F1 = lu.solve(F)
                ti += a*dt
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
                # print("ti",ti,"iB",iBarrier, M,Sc,B)
                A0 = A0Full.copy()
                A1 = A1Full.copy()
                A2 = A2Full.copy()
                BC = lil_matrix((L*M,L*M))
                updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
                A = A0+A1+A2
                Li = I+a*A+BC #FIXME compute A from 0, then update rows according to BC as iBarrier moves!
                lu = sla.splu(Li)
                updatePayoffBoundary(F1, Sc, B,iBarrier,M,L)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
    elif method == "O4":
        A = A0+A1+A2
        # a1 = 1.0/(6 - 2*math.sqrt(6))
        # a2 = 1.0/(2*(3+math.sqrt(6)))
        # lu1 = sla.splu(I + a1*A+BC)
        # lu2 = sla.splu(I + a2*A+BC)
        Asq = A*A
        Li0 = I+A+0.5*Asq+1.0/6*A*Asq
        lu0 = sla.splu(Li0+BC)
        lu = sla.splu(I+0.5*A+1.0/12*Asq+BC)
        #F0 = F - A*F + 0.5*A*A*F - 1.0/6* A*A*A*F
        #F1 = F0 -  A*F0 + 0.5*A*A*F0 - 1.0/6* A*A*A*F0# A*F0 + 0.5*A*(I-A/3)*(A*F0)

        updatePayoffBoundary(F, Sc, B,iBarrier,M,L)
        F0 = lu0.solve(F)
        updatePayoffBoundary(F0, Sc, B,iBarrier,M,L)
        F1 = lu0.solve(F0)
        F = F1
        for i in range(N-2):
            Fr= F-0.5*A*(F - 1.0/6*A*F)
            updatePayoffBoundary(Fr, S, B,iBarrier,M,L)
            # F1 = lu2.solve(Fr)
            # updatePayoffBoundary(F1, S, B,iBarrier,M,L)
            F = lu.solve(Fr)

    else:
        if B == 0:
            A0 = A0Full.copy()
            A1 = A1Full.copy()
            A2 = A2Full.copy()
            BC = lil_matrix((L*M,L*M))
            updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
            A = A0+A1+A2
            Li = I+A+BC
            lu = sla.splu(Li)
            for i in range(N):
                updatePayoffBoundary(F,Sc,B,iBarrier,M,L)
                F = lu.solve(F)
        else:
            for i in range(N):
                ti = T*(N-i-1)/N
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                iBarrier = np.searchsorted(Sc,B)  #S[i-1]<B<=S[i]
                # print("ti",ti,"iB",iBarrier, M,Sc,B)
                A0 = A0Full.copy()
                A1 = A1Full.copy()
                A2 = A2Full.copy()
                BC = lil_matrix((L*M,L*M))
                updateSystemBoundary(L, M, iBarrier, Sc, B, A0, A1, A2, BC)
                A = A0+A1+A2
                Li = I+A+BC
                lu = sla.splu(Li)
                updatePayoffBoundary(F,Sc,B,iBarrier,M,L)
                F = lu.solve(F)
    end=time.time()

    #F[50+4*M]
    #S0=101.52
    Payoff = F.reshape(L,M)
    #print("Payoff V=0",Payoff[0])
    jv0 = np.searchsorted(V,v0)
    #print("Payoff V=V0",V[jv0])
    #for (si,pi) in zip(S, Payoff[jv0]):
    #    print(si, pi)
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
    Payoffi = interpolate.RectBivariateSpline(V,X,Payoff,kx=3,ky=3,s=0)
    maxError = 0.0
#    Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    #print("spot method n m l price delta gamma error")
    #Vp = [(Payoffi(v0,x,dy=2)[0][0]-Payoffi(v0,x,dy=1)[0][0])*np.exp(-2*(x)) for x in X]
    #for Si, Vi in zip(S,Vp):
    #    print(Si, "HV-Euler", Vi)
    #plt.grid(True)
    ## plt.plot(S, [Payoffi(v0,math.log(Si),dy=1)[0][0]/(Si) for Si in S])
    #plt.plot(S, [Payoffi(v0,math.log(Si),dy=2)[0][0]/(Si*Si)-Payoffi(v0,math.log(Si),dy=1)[0][0]/(Si*Si) for Si in S])
    ## #plt.plot(V,Payoff[:][istrike]) d (dV/dx dx/ds) /dx dx/ds = d ( 1/)
    ## plt.yscale('symlog',linthreshy=1e-6)
    #plt.show()

    for spot,refPrice in zip(spotArray,priceArray):
        xspot = math.log(spot)
        price = Payoffi(v0,xspot)[0][0]
        delta = Payoffi(v0,xspot,dy=1)[0][0]
        gamma = Payoffi(v0,xspot,dy=2)[0][0]
        error = price -refPrice
        if abs(error) > maxError:
            maxError = abs(error)
        if B==0:
            print(spot,method,N,M,L, price, delta,gamma,error,end-start)
    if not B==0:
        print(method,N,M,L,Payoffi(v0,math.log(K))[0][0],end-start)
    # else:
    #     print(method,N,M,L,maxError,end-start)


def craigSneydBanded(N, F, I, A0, A1, A2,A1tri,A2tri, indices, indicesInv):
    a = 0.5
    A1tri*=a
    A1tri[1,:]+=1
    A2tri*=a
    A2tri[1,:]+=1

    for i in range(N):
        Y0 = (I-A0-A1-A2)*F #explicit
        #updatePayoffExplicit(Y0, S, B, iBarrier, M,L)
        Y0r = Y0+a*A1*F
        Y1 = la.solve_banded((1, 1), A1tri, Y0r, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y1r = Y1+a*A2*F
        Y1t = Y1r[indicesInv]
        Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y2 = Y2t[indices]
        Y0t = Y0 - 0.5*(A0*(Y2-F))
        Y0r = Y0t+a*A1*F
        Y1 = la.solve_banded((1, 1), A1tri, Y0r, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y1r = Y1+a*A2*F
        Y1t = Y1r[indicesInv]
        Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        F = Y2t[indices]
    return F

def hundsdorferVerwerBanded(N, F, I, A0, A1, A2,A1tri,A2tri, indices, indicesInv):
    a = 0.5+math.sqrt(3)/6
    A1tri*=a
    A1tri[1,:]+=1
    # A1tri[0]+= BCtri[0]
    # A1tri[1]+= BCtri[1]
    # A1tri[-1]+= BCtri[-1]
    A2tri*=a
    A2tri[1,:]+=1
    # A2tri[0] += BC2tri[0]
    # A2tri[1] += BC2tri[1]
    # A2tri[-1] += BC2tri[-1]
    for i in range(N):
        Y0 = (I-A0-A1-A2)*F #explicit
        Y0 = Y0+a*A1*F
        #Y0 = (I-A0-(1-a)*A1-A2)*F #explicit
        Y1 = la.solve_banded((1, 1), A1tri, Y0, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y1 = Y1+a*A2*F
        Y1t = Y1[indicesInv]
        Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y2 = Y2t[indices]
        Y0 = F-0.5*(A0+A1+A2)*(F+Y2)
        Y0 = Y0+a*A1*Y2
        #Y0 = F-0.5*(A0+A1+A2)*(F+Y2)+a*A1*Y2
        Y1 = la.solve_banded((1, 1), A1tri, Y0, overwrite_ab=False,overwrite_b=True, check_finite=False)
        Y1 = Y1+a*A2*Y2
        Y1t = Y1[indicesInv]
        Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        F = Y2t[indices]
    return F

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

@jit(nopython=True)
def updatePayoffBoundaryTrans(F, S, B, iBarrierList, M,L):
    if not B == 0:
        for j in range(L):
            iBarrier = iBarrierList[j]  #S[i-1]<B<=S[i]
            for ib in range(iBarrier):
                F[ib+j*M] = 0

@jit(nopython=True)
def updatePayoffExplicitTrans(F, S, B, iBarrierList, M,L):
    # Si-B *  Vim + Vi * B-Sim  =0
    if not B == 0:
        for j in range(L):
            iBarrier =iBarrierList[j]  #S[i-1]<B<=S[i]
            F[j*M:(iBarrier-1 +j*M)] = 0
            F[iBarrier-1 +j*M] = F[iBarrier + j*M] * (S[iBarrier-1]-B)/(S[iBarrier]-B)

#@jit(nopython=True)
def createSystemTrans(useExponentialFitting,B,iBarrierList,S,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L):
    upwindingThreshold = 10.0
    F = np.array(F0,copy=True)
    dt = -T/N
    A1 = lil_matrix((L*M,L*M))
    A2 = lil_matrix((L*M,L*M))
    BC = lil_matrix((L*M,L*M))

    #boundary conditions, 0,0, 0,L-1, M-1,0, M-1,L-1.
    if B == 0:
        i=0
        j=0
        A1[i+j*M,(i+1)+j*M] += dt*((r-q)/(hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*(-(r-q)/hm)
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        #A[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        i=0
        j=L-1
        A1[i+j*M,(i+1)+j*M] += dt*((r-q)/hm)
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*(-(r-q)/hm)
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
    else:
        j=0
        iBarrier = iBarrierList[j] #S[i-1]<B<=S[i]
        BC[iBarrier-1+j*M,iBarrier-1+j*M]=(S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
        BC[iBarrier-1+j*M,iBarrier+j*M]=(B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])
        # for i in range(iBarrier-1):
        #     A[i+j*M,i+(j)*M]=0.0
        j=L-1
        iBarrier = iBarrierList[j]  #S[i-1]<B<=S[i]
        BC[iBarrier-1+j*M,iBarrier-1+j*M]=(S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
        BC[iBarrier-1+j*M,iBarrier+j*M]=(B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])

    #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    i=M-1
    j=L-1
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)/(hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    i=M-1
    j=0
    A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(hm))
    A1[i+j*M,i+j*M] += dt*(-r*0.5)
    A1[i+j*M,(i)+j*M] += dt*((r-q)/(hm))
    A2[i+j*M,i+j*M] += dt*(-r*0.5)
    #A[i+j*M,i+(j+1)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    #A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))

    #boundary conditions j=0,L-1.
    j=0
    iBarrier = 1
    if not B == 0:
        iBarrier = iBarrierList[j]  #S[i-1]<B<=S[i]
    for i in range(iBarrier,M-1):
            #svj = sigma*sigma*V[j]/(JV[j])
            #driftj = kappa*(theta-V[j])
        svi = V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
        drifti = (r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
        if useExponentialFitting:
            if abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                #svi = svi + 0.5*abs(drifti)*hm

        A1[i+j*M,(i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
        A1[i+j*M,i+j*M] += dt*(-svi/(hm*hm)-r*0.5)
        A1[i+j*M,(i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
        A2[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))

    j=L-1
    iBarrier = 1
    if not B == 0:
        iBarrier = iBarrierList[j]  #S[i-1]<B<=S[i]
    for i in range(iBarrier,M-1):
        svi = V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
        drifti = (r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
        if useExponentialFitting:
            if abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
                # svi = svi + 0.5*abs(drifti)*hm
        A1[i+j*M,(i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
        A1[i+j*M,i+j*M] += dt*(-svi/(hm*hm)-r*0.5)
        A1[i+j*M,(i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
        A2[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        A2[i+j*M,i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))

    for j in range(1,L-1):
        #boundary conditions i=0,M-1.
        iBarrier = 1
        if B == 0:
            i=0
            A1[i+j*M,(i+1)+j*M] += dt*((r-q)/(hm))
            A1[i+j*M,i+j*M] += dt*(-r*0.5)
            A1[i+j*M,(i)+j*M] += dt*(-(r-q)/(hm))
            A2[i+j*M,i+j*M] += dt*(-r*0.5)
        else:
            iBarrier = iBarrierList[j]  #S[i-1]<B<=S[i]
            BC[iBarrier-1+j*M,iBarrier-1+j*M]=(S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
            BC[iBarrier-1+j*M,iBarrier+j*M]=(B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])

        #A2[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl*2)) #makes it  explode!
        #A2[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl*2))
        i=M-1
        A1[i+j*M,(i-1)+j*M] += dt*(-(r-q)/(hm))
        A1[i+j*M,i+j*M] += dt*(-r*0.5)
        A1[i+j*M,(i)+j*M] += dt*((r-q)/(hm))
        A2[i+j*M,i+j*M] += dt*(-r*0.5)
        #A2[i+j*M,i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(2*JV[j]*hl))
        #A2[i+j*M,i+(j+1)*M] += dt*(kappa*(theta-V[j])/(2*JV[j]*hl))
        for i in range(iBarrier,M-1):
            svj = sigma*sigma*V[j]/(JV[j])
            driftj = kappa*(theta-V[j])
            svi = V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
            drifti = (r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
            if useExponentialFitting:
                if abs(drifti*hm/svi) > upwindingThreshold:
                    svi = drifti*hm/math.tanh(drifti*hm/svi)
                    # svi = svi + 0.5*abs(drifti)*hm
                if driftj != 0 and abs(driftj*hl/svj) > upwindingThreshold:
                    # svj = svj + 0.5*abs(driftj)*hl
                    svj = driftj*hl/math.tanh(driftj*hl/svj)

            A1[i+j*M,(i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
            A1[i+j*M,i+j*M] += dt*(-(svi)/(hm*hm)-r*0.5)
            A1[i+j*M,(i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
            A2[i+j*M,i+(j+1)*M] += dt*(0.5*svj/(JVm[j+1]*hl*hl)+driftj/(2*JV[j]*hl))
            A2[i+j*M,i+j*M] += dt*(-r*0.5-0.5*svj/(hl*hl)*(1.0/JVm[j+1]+1.0/JVm[j]))
            A2[i+j*M,i+(j-1)*M] += dt*(0.5*svj/(JVm[j]*hl*hl)-driftj/(2*JV[j]*hl))


    A1tri, A2tri, indices, indicesInv = createTridiagonalIndices(M,L)
    # for i in range(M):
    #     for j in range(L):
    #         #A2tri[1,j+i*L] = A2[i+j*M,i+j*M]
    #         #if j < L-1:
    #         #    A2tri[-1,j+i*L] = A2[i+(j+1)*M,i+(j)*M]
    #         if j >0:
    #             A2tri[0,j+i*L] = A2[i+(j-1)*M,i+(j)*M]

    A0 = 0.0
    return F,A0,A1,A2,BC,A1tri, A2tri, indices, indicesInv

def createTridiagonalIndices(M,L):
    A1tri = np.zeros((3,M*L))
    A2tri = np.zeros((3,M*L))
    indices = np.zeros(M*L, dtype=int)
    indicesInv = np.zeros(M*L, dtype=int)
    for i in range(M):
        #for j in range(i,L):
        #    F2t[i+j*M] = F2[j+i*M] #numpy reshape 2d array? L,M -> M,L needs to be done at each iter.
        #    F2t[j+i*M] = F2[i+j*M]
        #    other fast way: create index list and then F2t = F2[indexList]
        #    or nd.array.flatten C/R alternatively.
        #A2t[i+j*M,:] = A2[j+i*M,:] #Ft will have reverse rows
        for j in range(L):
            indices[i+j*M]=j+i*L
            indicesInv[j+i*L] = i+j*M
    return A1tri, A2tri, indices, indicesInv

def objectivePayoff(K, lnK, hm, alpha, vj, xij, v):
    #eterm = math.exp(xij-alpha*(vj-v))
    #return np.maximum(eterm-K,0)
    if xij < -hm+alpha*(vj-v)+lnK:
        return 0.0
    elif xij > hm+alpha*(vj-v)+lnK:
        return -K + 2*(math.cosh(hm)-1)*math.exp(xij+alpha*(v-vj))/(hm*hm)
    else:
        eterm = math.exp(xij-alpha*(vj-v))
        y0 = -lnK+xij-alpha*(vj-v)
        if xij > alpha*(vj-v)+lnK: # i1 = int_0^h eterm*e(-y)-K , i2=int_-h,0 eterm(e(-y))-K
            eY0 = K # eterm*math.exp(-y0)
            i1 = (hm-1)/(hm*hm)*eterm + (eY0*(y0+1-hm) + K*y0*y0/2-K*hm*y0)/(hm*hm)
            i2 = -K/2 + eterm*(math.exp(hm)-(hm+1))/(hm*hm)
            return i1+i2
        else:
            i1 = 0.0
            eY0 = K # eterm*math.exp(-y0)
            i2 = eterm*math.exp(hm)/(hm*hm)-K/2*(1+y0*y0/(hm*hm)+2*y0/hm)-2*eY0*(y0+hm+1)/(2*hm*hm)
            return i1+i2

#@jit(nopython=True)
def smoothKreissTrans(L,M,X,V,hm,alpha,K, sign, cFunc,F0):
    for j in range(1,L-1):
        for i in range(M):
            xij = X[i]
            vij = V[j]
            hmt = hm
            hlt = (V[j+1]-V[j-1])/2
            payoff1 = lambda v,y: max(sign*(cFunc.evaluate(math.exp(xij-y-alpha*(vij-v)))-K),0)*(1-abs(y)/hmt)*(1-abs(v)/hlt)
            #payoff1 = np.vectorize(payoff1)
            value1 = integrate.dblquad(payoff1, -hmt, hmt, lambda x: -hlt, lambda x: hlt)
            F0[i+j*M] = value1[0]/(hlt*hmt)

def priceCallTransformed(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K,B, N, M, L):
    isCall = False
    method = "PR" # "LODLS" "CS" "LS","CN","DO"
    damping = "Euler" #"None" "One", "Euler"
    smoothing = "None" #"None","Kreiss"
    useVLinear = False
    useExponentialFitting=True
    alpha = -rho/sigma

    epsilon = 1e-3
    dChi = 4*kappa*theta/(sigma*sigma)
    chiN = 4*kappa*math.exp(-kappa*T)/(sigma*sigma*(1-math.exp(-kappa*T)))
    vmax = ncx2.ppf((1-epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    vmin = ncx2.ppf((epsilon),dChi,v0*chiN)*math.exp(-kappa*T)/chiN
    # print("vmax",vmin,vmax, 10*v0)
    vmin=max(1e-3,vmin) #Peclet explodes at V=0!
    #vmax=10*v0#0.28
    V = np.linspace(vmin,vmax,L)
    hl = V[1]-V[0]
    JV=np.ones(L)
    JVm=np.ones(L)

    if not useVLinear:
        vscale = v0*2
        u = np.linspace(0,1,L) #1e-4,math.sqrt(vmax),L)  #ideally, concentrated around v0: V=sinh((w-w0)/c). w unif
        c1 = math.asinh((vmin-v0)/vscale)
        c2 = math.asinh((vmax-v0)/vscale)
        V = v0 + vscale*np.sinh((c2-c1)*u+c1)
        hl = u[1]-u[0]
        JV = vscale*(c2-c1)* np.cosh((c2-c1)*u+c1)
        JVm = vscale*(c2-c1)* np.cosh((c2-c1)*(u-hl/2)+c1)

#case 1:
    #Xspan = 6*math.sqrt(theta*T)
#case 2: Xspan = 10*math.sqrt(theta*T)
#5ok, 6 breaks on case 1!? 800x200
    Xspan = 4*math.sqrt(theta*T) #max(4*math.sqrt(theta*T),(0.5*math.sqrt(v0*T)+abs(alpha*vmax))) #+abs(r-q)*T
    Kinv = cFunc.solve(K)
    Xmin = math.log(Kinv)-Xspan+alpha*v0
    Xmax = math.log(Kinv)+Xspan+alpha*v0
    #print("Xmin",Xmin,"Xmax",Xmax)
    X = np.linspace(Xmin,Xmax,M)
    hm = X[1]-X[0]

    #V
    # pecletL = np.zeros(L)
    # pecletM = np.zeros(L)
    # sCoeff = V*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
    # dCoeff = r-q-0.5*V+kappa*(theta-V)*alpha
    # pecletM = dCoeff/sCoeff*hm
    # sCoeff = sigma*sigma*V/(JV*JVm)
    # dCoeff = kappa*(theta-V)/JV
    # pecletL = dCoeff/sCoeff*hl
    # print("PecletL",pecletL)
    # print("PecletM",pecletM)

    F0 = np.zeros(M*L)
    S = np.zeros(M*L)
    lnK = math.log(K)
    sign = 1
    if not isCall:
        sign = -1

    for j in range(L):
        for i in range(M):
            S[i+j*M]=np.exp(X[i]-alpha*V[j])

    Sc = np.array([cFunc.evaluate(T,Si) for Si in S])
    F0 = np.maximum(sign*(Sc-K),0)

    if smoothing == "Kreiss":
        smoothKreissTrans(L,M,X,V,hm,alpha, K,sign,cFunc,F0)
        #payoff = lambda x: np.maximum(np.exp(x-alpha*V[j])-K,0)
        #value = integrate.quadrature( payoff, X[iStrike]-hm/2, X[iStrike]+hm/2)
        # int_xK^A exp(x-al*v)-K   = exp(x-al*v)-K-K*(x-xK)
        #xm = X[iStrike]+hm/2
        #value = np.exp(xm-alpha*V[j])-K-K*(xm-xk)  #int vm
        #if j > 0 and j < L-1:
        #    value = (np.exp(xm-alpha*(V[j-1]+V[j])/2)-np.exp(xm-alpha*(V[j+1]+V[j])/2))/(alpha*(V[j+1]-V[j-1])/2)-K-K*(xm-xk)
        #F0[iStrike+j*M] = value/hm

    # for j in range(L):
    #     plt.grid(True)
    #     plt.plot(X, F0[j*M:j*M+M])
    # plt.show()

    iBarrierList = np.zeros(L,dtype='int')
    for j in range(L):
        iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]

    #print("iBarrierList",iBarrierList)
    start = time.time()
    F, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
    end = time.time()
    #print("setup matrices",end-start)
    A1 = A1.tocsc()
    A2 = A2.tocsc()
    A1tri[1,:] = A1.diagonal(k=0)
    A1tri[-1,:-1] = A1.diagonal(k=-1)
    A1tri[0,1:] = A1.diagonal(k=1)
    A2tri[1,:] = A2.diagonal(k=0)[indicesInv]
    A2i = A2[:,indicesInv]
    A2i = A2i[indicesInv,:]
    #print("size",A2i.shape)
    A2tri[-1,:-1] = A2i.diagonal(k=-1)
    A2tri[0,1:] = A2i.diagonal(k=1)
    #  ab = np.ones((3, n - 1), dtype=float)
    #ab[0,0] = 0 # Because top row is upper diag with one less elem
    #ab[1, :] = 4
    #ab[-1,-1] = 0 # Because bottom row is lower diag with one less elem
    #
    end = time.time()
    #print("setup tri",end-start)
    I = identity(M*L,format="csc")
    start=time.time()
    if damping == "Euler":
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A+BC
        lu = sla.splu(Li)
        updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
        F = lu.solve(F)
        updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
        F = lu.solve(F)
        N -= 1

    if method == "EU":
        A = A0+A1+A2
        Li = I+A+BC
        lu = sla.splu(Li)
        for i in range(N):
            updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
            F=lu.solve(F)
    elif method == "CS":
        a = 0.5
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)

        # A1tri*=a
        # A1tri[1,:]+=1
        # A2tri*=a
        # A2tri[1,:]+=1
        # def solveCS(a,F):
        #     for i in range(N):
        #         Y0 = F-(A0+A1+A2)*F #explicit
        #         Y1 = la.solve_banded((1, 1), A1tri, Y0+a*(A1*F), overwrite_ab=False,overwrite_b=True, check_finite=False)
        #         Y1t = Y1+a*(A2*F)
        #         Y1t = Y1t[indicesInv]
        #         Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        #         Y2 = Y2t[indices]
        #         Y0 = Y2- 0.5*(A0*Y2-A0*F)
        #         Y1 = la.solve_banded((1, 1), A1tri, Y0+a*(A1*F), overwrite_ab=False,overwrite_b=True, check_finite=False)
        #         Y1t = Y1+a*(A2*F)
        #         Y1t = Y1t[indicesInv]
        #         Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        #         F = Y2t[indices]

        for i in range(N):
            #updatePayoffExplicitTrans(F,S,B,iBarrierList,M,L)
            Y0 = (I-A0-A1-A2)*F #explicit
            #updatePayoffExplicitTrans(Y0,S,B,iBarrierList,M,L)
            Y0r = Y0+a*A1*F
            updatePayoffBoundaryTrans(Y0r,Sc,B,iBarrierList,M,L)
            Y1 = lu1.solve(Y0r)
            Y1r = Y1+a*A2*F
            updatePayoffBoundaryTrans(Y1r,Sc,B,iBarrierList,M,L)
            Y2 = lu2.solve(Y1r)
            Y0t = Y0 - 0.5*(A0*Y2-A0*F)
            Y0r = Y0t + a*A1*F
            updatePayoffBoundaryTrans(Y0r,Sc,B,iBarrierList,M,L)
            Y1t = lu1.solve(Y0r)
            Y1r = Y1t+a*A2*F
            updatePayoffBoundaryTrans(Y1r,Sc,B,iBarrierList,M,L)
            Y2t = lu2.solve(Y1r)
            F = Y2t

    elif method == "DO":
        a = 0.5
        lu1 = sla.splu(I+a*A1+BC)
        lu2 = sla.splu(I+a*A2+BC)
        for i in range(N):
            updatePayoffExplicitTrans(Y0r,Sc,B,iBarrierList,M,L)
            Y0 = F-(A0+A1+A2)*F #explicit
            Y0r = Y0+a*(A1*F)
            updatePayoffBoundaryTrans(Y0r,Sc,B,iBarrierList,M,L)
            Y1 = lu1.solve(Y0r)
            Y1r = Y1+a*(A2*F)
            updatePayoffBoundaryTrans(Y1r,Sc,B,iBarrierList,M,L)
            Y2 = lu2.solve(Y1r)
            F = Y2
        # A1tri*=a
        # A1tri[1,:]+=1
        # A2tri*=a
        # A2tri[1,:]+=1
        # for i in range(N):
        #     Y0 = F-(A0+A1+A2)*F #explicit
        #     Y1 = la.solve_banded((1, 1), A1tri, Y0+a*(A1*F), overwrite_ab=False,overwrite_b=True, check_finite=False)
        #     Y1t = Y1+a*(A2*F)
        #     Y1t = Y1t[indicesInv]
        #     Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
        #     Y2 = Y2t[indices]
        #     #print("indices",indices)
        #     #print("indicesInv",indicesInv)
        #     #print("Zero?")
        #     #print(A2*Y2-(Y1+a*A2*F))
        #     F = Y2
        #     #raise Exception("print")

    elif method == "PR": #peaceman-rachford strikwerda
        a = 0.5
        if B == 0:
            A1tri*=a
            A1tri[1,:]+=1
            # A1tri[0]+= BCtri[0]
            # A1tri[1]+= BCtri[1]
            # A1tri[-1]+= BCtri[-1]
            A2tri*=a
            A2tri[1,:]+=1
            # A2tri[0] += BC2tri[0]
            # A2tri[1] += BC2tri[1]
            # A2tri[-1] += BC2tri[-1]
            if damping == "One":
                Y1 = la.solve_banded((1, 1), A1tri, F, overwrite_ab=False,overwrite_b=False, check_finite=False)
                Y1t = Y1[indicesInv]
                Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=False, check_finite=False)
                Y2t = la.solve_banded((1, 1), A2tri, Y2t, overwrite_ab=False,overwrite_b=False, check_finite=False)
                Y2 = Y2t[indices]
                Y1 = la.solve_banded((1, 1), A1tri, Y2, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F = Y1
                N -= 1
            #
            # #A2triM = dia_matrix((A2tri,[1,0,-1]),shape=(M*L,M*L))
            for i in range(N):
                #updatePayoffExplicitTrans(F,S,B,iBarrierList,M,L)
                Y0 =  F-a*(A2*F)
                Y1 = la.solve_banded((1, 1), A1tri,Y0, overwrite_ab=False,overwrite_b=True, check_finite=False)
                Y1t = Y1-a*(A1*Y1)
                Y1t = Y1t[indicesInv]
                Y2t = la.solve_banded((1, 1), A2tri, Y1t, overwrite_ab=False,overwrite_b=True, check_finite=False)
                F = Y2t[indices]
        else:
            #second system is not tridiag because of boundary
            ti=T
            dt = 1.0/N
            updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L) #FIXME ideally mirror F[ib-1] from F[ib]           
            if damping == "One":
                ti -= dt*0.5
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                lu1 = sla.splu(I+a*A1+BC)
                F = lu1.solve(F)
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                lu2 = sla.splu(I+a*A2+BC)
                F = lu2.solve(F)
                ti -= dt*0.5
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                lu2 = sla.splu(I+a*A2+BC)
                F = lu2.solve(F)
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                lu1 = sla.splu(I+a*A1+BC)
                F = lu1.solve(F)
                N-=1

            for i in range(N): #PROBLEM: BOUNDARY is NEUMAN, NOT DIRICHLET=>breaks
                # Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                # for j in range(L):
                #     iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                # Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                ti -= dt*0.5
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                # updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L)
                Y0 =  (I)*F-a*(A2*F)
                updatePayoffBoundaryTrans(Y0,Sc,B,iBarrierList,M,L)
                lu1 = sla.splu(I+a*A1+BC)
                Y1 = lu1.solve(Y0)                
                Y1t = (I)*Y1-a*(A1*Y1)
                # updatePayoffExplicitTrans(Y1t,Sc,B,iBarrierList,M,L) #FIXME ideally mirror F[ib-1] from F[ib]           
                updatePayoffBoundaryTrans(Y1t,Sc,B,iBarrierList,M,L)
                lu2 = sla.splu(I+a*A2+BC)
                Y2 = lu2.solve(Y1t)
                F = Y2
                ti -= dt*0.5
                # Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                # for j in range(L):
                #     iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                # Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                # updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L)

    elif method == "CN":
        A = A0+A1+A2
        a = 0.5
        Li = I+a*A+BC
        Le = I-(1-a)*A
        lu = sla.splu(Li)
        for i in range(N):
            #updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L)
            Y0 = Le*F
            updatePayoffBoundaryTrans(Y0,Sc,B,iBarrierList,M,L)
            F = lu.solve(Y0)
    elif method == "PRLS":
        a = 1 - math.sqrt(2)/2
        #a*=0.5
        Li = I+a*A1
        lu = sla.splu(Li)
        lu2 = sla.splu(I+a*(A0+A2))

        for i in range(N): #TODO try TR like PR (A1 then A2) then BDF2 on Y2,A1 and BDF2 on Y2,A2.
            F1 = lu.solve(F)
            F1t = lu2.solve(F1)
            F1t = (1+math.sqrt(2))*F1t - math.sqrt(2)*F1
            F2 = lu2.solve(F1t)
            F2t = lu.solve(F2)
            F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F2
    elif method == "SLSB":
        a = 1 - math.sqrt(2)/2
        a*=0.5
        Li = I+a*A1
        lu = sla.splu(Li)
        lu2 = sla.splu(I+a*(A0+A2))

        for i in range(N):
            F1 = lu2.solve(F)
            F1t = lu2.solve(F1)
            F = (1+math.sqrt(2))*F1t - math.sqrt(2)*F1
            F2 = lu.solve(F)
            F2t = lu.solve(F2)#strang splitting?
            F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F2
            F1 = lu.solve(F)
            F1t = lu.solve(F1)
            F = (1+math.sqrt(2))*F1t - math.sqrt(2)*F1
            F2 = lu2.solve(F)
            F2t = lu2.solve(F2)
            F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F2
    elif method == "Glowinski":
        a = 1 - math.sqrt(2)/2
        a2 = 1 - 2*a
        lu = sla.splu(I+a*A2+BC)
        lu2 = sla.splu(I+a2*(A0+A1)+BC)

        for i in range(N):
            Ft = F - a*A1*F
            updatePayoffBoundaryTrans(Ft,Sc,B,iBarrierList,M,L)
            F1 = lu.solve(Ft)
            F1t = F1 - a2*A2*F1
            updatePayoffBoundaryTrans(F1t,Sc,B,iBarrierList,M,L)
            F2 = lu2.solve(F1t)
            F2t = F2 - a*A1*F2
            updatePayoffBoundaryTrans(F2t,Sc,B,iBarrierList,M,L)
            F = lu.solve(F2t)
    elif method == "SLS":
        a = 1 - math.sqrt(2)/2
        if B == 0:
            A1tri*= a*0.5
            A1tri[1,:]+=1
            # A1tri[0]+= BCtri[0]
            # A1tri[1]+= BCtri[1]
            # A1tri[-1]+= BCtri[-1]
            A2tri*=a
            A2tri[1,:]+=1
            # A2tri[0] += BC2tri[0]
            # A2tri[1] += BC2tri[1]
            # A2tri[-1] += BC2tri[-1]
            for i in range(N):
                F1 = la.solve_banded((1, 1), A1tri, F, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F1b = la.solve_banded((1, 1), A1tri, F1, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F = (1+math.sqrt(2))*F1b - math.sqrt(2)*F1
                Ft = F[indicesInv]
                F2t = la.solve_banded((1, 1), A2tri, Ft, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F2bt = la.solve_banded((1, 1), A2tri, F2t, overwrite_ab=False,overwrite_b=False, check_finite=False)
                Ft = (1+math.sqrt(2))*F2bt - math.sqrt(2)*F2t
                F = Ft[indices]
                F1 = la.solve_banded((1, 1), A1tri, F, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F1b = la.solve_banded((1, 1), A1tri, F1, overwrite_ab=False,overwrite_b=False, check_finite=False)
                F = (1+math.sqrt(2))*F1b - math.sqrt(2)*F1
        else:
            dt = T/N
            for i in range(N):
                ti = T - i*dt
                Sc = np.array([cFunc.evaluate(ti+0.5*dt, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]                    
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
               
                luh = sla.splu(I+0.5*a*A1+BC)
                lu2 = sla.splu(I+a*(A0+A2)+BC)

                F1 = luh.solve(F)
                updatePayoffBoundaryTrans(F1,Sc,B,iBarrierList,M,L)
                F1t = luh.solve(F1)
                F = (1+math.sqrt(2))*F1t - math.sqrt(2)*F1

                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                F2 = lu2.solve(F)
                updatePayoffBoundaryTrans(F2,Sc,B,iBarrierList,M,L)
                F2t = lu2.solve(F2)#strang splitting?
                F = (1+math.sqrt(2))*F2t - math.sqrt(2)*F2

                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                F1 = luh.solve(F)
                updatePayoffBoundaryTrans(F1,Sc,B,iBarrierList,M,L)
                F1t = luh.solve(F1)
                F = (1+math.sqrt(2))*F1t - math.sqrt(2)*F1

    else: #if method =="LS":
        a = 1 - math.sqrt(2)/2
        dt = -T/N
        if B==0:
            A = A0+A1+A2
            Li = I+a*A+BC
            lu = sla.splu(Li) #ilu(Li,drop_tol=1e-10,fill_factor=1000)

            for i in range(N):
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                F1 = lu.solve(F)
                updatePayoffBoundaryTrans(F1,Sc,B,iBarrierList,M,L)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1
                #F = np.maximum(F,0)
        else:
            for i in range(N):
                ti = T*(N-i)/N
                ti = ti+a*dt
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                updatePayoffBoundaryTrans(F,Sc,B,iBarrierList,M,L)
                A = A0+A1+A2
                Li = I+a*A+BC
                lu = sla.splu(Li) #ilu(Li,drop_tol=1e-10,fill_factor=1000)
                F1 = lu.solve(F)
                ti = ti+a*dt
                Sc = np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j] = np.searchsorted(Sc[j*M:(j+1)*M],B)  #S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv = createSystemTrans(useExponentialFitting,B,iBarrierList,Sc,F0, V, JV, JVm, r,q,kappa,theta,rho,sigma,alpha, hm,hl,T, N, M, L)
                updatePayoffBoundaryTrans(F1,Sc,B,iBarrierList,M,L)
                A = A0+A1+A2
                Li = I+a*A+BC
                lu = sla.splu(Li) #ilu(Li,drop_tol=1e-10,fill_factor=1000)
                F2 = lu.solve(F1)
                F = (1+math.sqrt(2))*F2 - math.sqrt(2)*F1

    end=time.time()

    #F[50+4*M]
    #S0=101.52
    Payoff = F.reshape(L,M)

    # print("Payoff V=0",Payoff[0])
    # jv0 = np.searchsorted(V,v0)
    # print("Payoff V=V0",V[jv0])
    # for (si,pi) in zip(S[:M], Payoff[jv0]):
    #     print(si, pi)
    #
    # # istrike =np.searchsorted(S,K)
    # # print("Payoff S=K",S[istrike])
    # # for (vi,pi) in zip(V, Payoff[:][istrike]):
    # #     print(vi, pi)
    # #plt.ion()
    # plt.grid(True)
    # plt.plot(S[iBarrier:iBarrier+30], Payoff[jv0][iBarrier:iBarrier+30])
    # #plt.plot(V,Payoff[:][istrike])
    # plt.yscale('symlog',linthreshy=1e-6)
    # plt.show()

    #Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    Payoffi = interpolate.RectBivariateSpline(V,X,Payoff,kx=3,ky=3,s=0)

    # Sp = np.exp(X-alpha*v0)
    # Vp = [(Payoffi(v0,x,dy=2)[0][0]-Payoffi(v0,x,dy=1)[0][0])*np.exp(-2*(x-alpha*v0)) for x in X]
    # for Si, Vi in zip(Sp,Vp):
    #    print(Si, "PR-Damped-S", Vi)
    #
    # plt.grid(True)
    # # plt.plot(np.exp(X-alpha*v0),[Payoffi(v0,x,dy=2)[0][0] for x in X])
    # plt.plot(Sp,Vp)
    # # z = z(y,v) = y - alpha*v, v= v => d/dy = d/dz*dz/dy
    # plt.show()
    maxError = 0.0

#    Payoffi = interpolate.interp2d(S,V,Payoff,kind='cubic')
    for spot,refPrice in zip(spotArray,priceArray):
        x0 = math.log(spot)+alpha*v0
        price = Payoffi(v0,x0)[0][0]
        delta = Payoffi(v0,x0,dy=1)[0][0]
        gamma = Payoffi(v0,x0,dy=2)[0][0]
        error = price -refPrice
        if abs(error) > maxError:
            maxError = abs(error)
        if B==0:
            print(spot,method,N,M,L, price, delta,gamma,error,end-start)
    if B == 0:
        pass #print(method,N,M,L,maxError,end-start)
    else:
        x0 = math.log(K)+alpha*v0
        print(method,N,M,L,Payoffi(v0,x0)[0][0],end-start)

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
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [0.4290429592804125, 0.5727996675731273, 0.7455984677403922, 0.9488855729391782, 1.1836198521834569, 1.4503166421285438, 1.7491038621459454, 2.079782505454696, 2.4418861283930053, 2.834736019523883, 3.257490337101448, 3.709186519701557, 4.188777097589518, 4.6951592762243415, 5.227198998513091, 5.7837501984978665, 6.363669958734282, 6.965830262856437, 7.589126920735202, 8.232486143930792, 8.894869093849636, 9.575277129770623, 10.272748751757314, 10.986365852615036, 11.715254013220457, 12.458577567319875, 13.215544738495424, 13.98540421747423, 14.767442110445812, 15.560982138391632, 16.36538729643898, 17.180051769091545, 18.004405483745735, 18.8379101967189, 19.68005854335592, 20.53036894075123, 21.388390582359417, 22.25369629176841, 23.12588767795124, 24.004578691901752, 24.889416575642677]
    M = 401 #X
    L = 101 #V
    Ms = [25, 51, 101, 201, 401]
    Ls = [12, 25, 51, 101, 201]
    Ms = [201]
    Ls= [31]
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
    L = 101 #V
    Ns = [2048, 1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
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
    B=0.0
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [4.126170747504533, 4.408743197301329, 4.70306357455405, 5.009202471608047, 5.327215893333642, 5.657145552450321, 5.999019203695557, 6.3528510118569015, 6.718641951722364, 7.096380233599666, 7.486041751584794, 7.887590552192177, 8.300979318221902, 8.726149865537172, 9.163033649989693, 9.611552278338717, 10.071618030216948, 10.543134388629074, 11.025996479014745, 11.520091740844437, 12.025300295511904, 12.54149551835306, 13.068544517640353, 13.606308624804461, 14.154643874270963, 14.713401467714998, 15.282428228751144, 15.861567038426507, 16.450657265344518, 17.04953517774978, 17.658034469027065, 18.2759861100527, 18.903219497330056, 19.539562310453945, 20.184840914482272, 20.838880779749626, 21.501506644797566, 22.17254294281439, 22.85181397102651, 23.539144197874872, 24.23435849148654]
    Ms = [25, 51, 101, 201, 401]
    Ls = [12, 25, 51, 101, 201]

    Ms = [51]
    Ls= [12]
    N = 32#s = [4,8,16,32,64,128] #timesteps
    for L,M in zip(Ls,Ms):
        priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, B, N, M, L)

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
    M = 101 #X
    L = 21 #V
    B=0
    Ns = [2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns.reverse()
    for N in Ns:
        priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, B, N, M, L)

def priceQLBarrierTime():
    kappa = 2.5
    theta = 0.04
    sigma = 0.66
    rho = -0.8
    r = 0.05
    q = 0.0
    v0=theta
    T=1.0
    K=100.0
    isCall = True
    spotArray = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
    priceArray = [4.126170747504533, 4.408743197301329, 4.70306357455405, 5.009202471608047, 5.327215893333642, 5.657145552450321, 5.999019203695557, 6.3528510118569015, 6.718641951722364, 7.096380233599666, 7.486041751584794, 7.887590552192177, 8.300979318221902, 8.726149865537172, 9.163033649989693, 9.611552278338717, 10.071618030216948, 10.543134388629074, 11.025996479014745, 11.520091740844437, 12.025300295511904, 12.54149551835306, 13.068544517640353, 13.606308624804461, 14.154643874270963, 14.713401467714998, 15.282428228751144, 15.861567038426507, 16.450657265344518, 17.04953517774978, 17.658034469027065, 18.2759861100527, 18.903219497330056, 19.539562310453945, 20.184840914482272, 20.838880779749626, 21.501506644797566, 22.17254294281439, 22.85181397102651, 23.539144197874872, 24.23435849148654]
    M = 101 #X
    L = 21 #V
    B=0.
    Ns = [2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns.reverse()
    for N in Ns:
        priceCall(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, K, B, N, M, L)

class IdentityFunction:
    def __init__(self):
        pass

    def evaluate(self, z):
        return z

class CollocationFunction:
    X = []
    A = []
    B = []
    C = []
    leftSlope = 0.0
    rightSlope = 0.0
    T = 0.0

    def __init__(self, X, A, B, C,leftSlope,rightSlope, T):
        self.X = X
        self.A = A
        self.B = B
        self.C = C
        self.leftSlope = leftSlope
        self.rightSlope = rightSlope
        self.T = T

    def evaluateSlice(self, z):
        if z <= self.X[0]:
            return self.leftSlope*(z-self.X[0]) + self.A[0]
        elif z >= self.X[-1]:
            return self.rightSlope*(z-self.X[-1])+self.A[-1]
        i = np.searchsorted(self.X,z)  # x[i-1]<z<=x[i]
        if i > 0:
            i -= 1
        h = z-self.X[i]
        return self.A[i] + h*(self.B[i]+h*self.C[i])

    def evaluate(self, t, z):
        return t/self.T * self.evaluateSlice(z) + (1.0-t/self.T)*z #linear interpolation between slice at t=0 and slice T.

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
    A=[0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
    B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941]
    C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
    X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
    leftSlope=0.8329310535215612
    rightSlope=0.2668764075068484
    #Absorption 0.001 0
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
    cFunc = CollocationFunction(X,A,B,C,leftSlope,rightSlope,T)
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

    M = 128   #X
    L = 128 #V
    B=0.8
    # Ns = [4096,2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns = [4096,1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
    # Ns = [72,60,12]
    Ns.reverse()
    for N in Ns:
        priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)

def priceSX5ESpace():
    #Spline 1e-5 pennalty
    A=[0.6287965835693049 ,0.8796805556963849 , 0.9548458991431029 ,0.9978807937190832 ,1.0432949917908245, 1.0951689975427406, 1.1780329537431, 1.2767467611605525]
    B=[0.846962887118158, 0.5006951388813219 ,1.3162296284270554, 0.764281474912235, 1.4312564546785838, 1.0765792448141005, 0.9264392665602718]
    C=[-0.46500629962499923, 4.928351101396242, -6.670948501034147, 8.061184212984527, -4.286695020953507, -0.907309913530479, -1.9936316682418205]
    X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
    leftSlope=0.846962887118158
    rightSlope=0.2666342520834516

    #Spline 1e-3 penalty
    # A=[0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
    # B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941]
    # C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
    # X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
    # leftSlope=0.8329310535215612
    # rightSlope=0.2668764075068484
    #Absorption 0.001 0
    cFunc = CollocationFunction(X,A,B,C,leftSlope,rightSlope)
    println("S=1 => X=",cFunc.solve(1.0))
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
    # priceArray = [0.07278065]
    # K=0.7
    # spotArray = [1.0]
    # priceArray = [0.30953450-0.3] #P = C- F-K
    # priceArray = [0.00960629]
    # K=1.4
    # spotArray = [1.0]
    # priceArray = [0.00015184+.4]
    # priceArray = [0.40015225]

    Ms= [8,12, 16,24, 32,48, 64, 96, 128,192, 256,512]  #X
    Ls = [8,12, 16,24, 32,48, 64, 96, 128,192, 256,512] #V
    L = 256
    B=0
    #Ns = [4096,2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    N = 64 #timesteps
    for L,M in zip(Ls,Ms):
    # for M in Ms:
        priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)

class PolyCollocationFunction:
    coeff = []

    def __init__(self, coeff):
        self.coeff = coeff

    def evaluate(self, z):
        return np.polyval(self.coeff,z)

    def solve(self, strike):
        c = self.coeff.copy()
        c[-1] -= strike
        allck = np.roots(c)
        #print("allck",allck)
        for ck in allck:
            if abs(ck.imag) < 1e-10:
                return ck.real
        raise Exception("no roots found in range", allck, strike, aa, bb, cc, i,self.X[i-1],self.X[i])

def pricePolySX5ETime():
    coeff = [-0.01969830242950278 ,0.9836590390856135 ,-2.127280418584288, 24.46758278682982 ,-68.69895549895567, 81.68521250909365 ,-44.40158377607094 ,9.096571378087397]
    coeff = [0.17074678852059158 ,0.824747250438463, 0.0071906167596872, 5.6862073468872206e-05]
    coeff.reverse()
    cFunc = PolyCollocationFunction(coeff)
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
    priceArray = [0.07211350]
    priceArray = [0.06937973] #call
    # K=0.7
    # spotArray = [1.0]
    # priceArray = [0.31095779]
    K=1.4
    spotArray = [1.0]
    priceArray = [0.39934721]
    M =64
    L = 201 #V
    B=0
    Ns = [2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns.reverse()
    for N in Ns:
        priceCallLog(spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)

def main():
    # priceAlbrecherSpace()
    # priceAlbrecherTime()
    #priceBloombergSpace()
    #priceBloombergTime()
    priceSX5ETime()

if __name__ =='__main__':
    main()
