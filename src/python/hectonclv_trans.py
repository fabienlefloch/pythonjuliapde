import numpy as np
import math
import time
from scipy.sparse import csc_matrix, lil_matrix, dia_matrix, identity, linalg as sla
from scipy import linalg as la
from scipy.stats import ncx2
from scipy import integrate
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def updatePayoffBoundaryTrans(F, S, B, iBarrierList, M, L):
    if not B == 0:
        for j in range(L):
            iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
            for ib in range(iBarrier):
                F[ib+j*M] = 0


def updatePayoffExplicitTrans(F, S, B, iBarrierList, M, L):
    # Si-B *  Vim + Vi * B-Sim  =0
    if not B == 0:
        for j in range(L):
            iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
            F[j*M:(iBarrier-1 + j*M)] = 0
            # F[iBarrier + j*M] * (S[iBarrier-1]-B)/(S[iBarrier]-B)
            F[iBarrier-1 + j*M] = 0


def createSystemTrans(useExponentialFitting, B, iBarrierList, S, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L):
    upwindingThreshold = 10.0
    F = np.array(F0, copy=True)
    dt = -T/N
    A1 = lil_matrix((L*M, L*M))
    A2 = lil_matrix((L*M, L*M))
    BC = lil_matrix((L*M, L*M))
    # boundary conditions, 0,0, 0,L-1, M-1,0, M-1,L-1.
    if B == 0:
        i = 0
        j = 0
        A1[i+j*M, (i+1)+j*M] += dt*((r-q)/(hm))
        A1[i+j*M, i+j*M] += dt*(-r*0.5)
        A1[i+j*M, (i)+j*M] += dt*(-(r-q)/hm)
        A2[i+j*M, i+j*M] += dt*(-r*0.5)
        # A[i+j*M,i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        # A[i+j*M,i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        i = 0
        j = L-1
        A1[i+j*M, (i+1)+j*M] += dt*((r-q)/hm)
        A1[i+j*M, i+j*M] += dt*(-r*0.5)
        A1[i+j*M, (i)+j*M] += dt*(-(r-q)/hm)
        A2[i+j*M, i+j*M] += dt*(-r*0.5)
    else:
        j = 0
        iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
        BC[iBarrier-1+j*M, iBarrier-1+j *   M] = (S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
        BC[iBarrier-1+j*M, iBarrier+j *   M] = (B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])
        j = L-1
        iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
        BC[iBarrier-1+j*M, iBarrier-1+j *   M] = (S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
        BC[iBarrier-1+j*M, iBarrier+j *  M] = (B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])
    i = M-1
    j = L-1
    A1[i+j*M, (i-1)+j*M] += dt*(-(r-q)/(hm))
    A1[i+j*M, i+j*M] += dt*(-r*0.5)
    A1[i+j*M, (i)+j*M] += dt*((r-q)/(hm))
    A2[i+j*M, i+j*M] += dt*(-r*0.5)
    i = M-1
    j = 0
    A1[i+j*M, (i-1)+j*M] += dt*(-(r-q)/(hm))
    A1[i+j*M, i+j*M] += dt*(-r*0.5)
    A1[i+j*M, (i)+j*M] += dt*((r-q)/(hm))
    A2[i+j*M, i+j*M] += dt*(-r*0.5)
    # boundary conditions j=0,L-1.
    j = 0
    iBarrier = 1
    if not B == 0:
        iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
    for i in range(iBarrier, M-1):
        svi = V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
        drifti = (r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
        if svi != 0 and useExponentialFitting:
            if abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
        A1[i+j*M, (i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
        A1[i+j*M, i+j*M] += dt*(-svi/(hm*hm)-r*0.5)
        A1[i+j*M, (i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
        A2[i+j*M, i+(j+1)*M] += dt*(+kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M, i+j*M] += dt*(-r*0.5)
        A2[i+j*M, i+(j)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
    j = L-1
    iBarrier = 1
    if not B == 0:
        iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
    for i in range(iBarrier, M-1):
        svi = V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
        drifti = (r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
        if useExponentialFitting:
            if abs(drifti*hm/svi) > upwindingThreshold:
                svi = drifti*hm/math.tanh(drifti*hm/svi)
        A1[i+j*M, (i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
        A1[i+j*M, i+j*M] += dt*(-svi/(hm*hm)-r*0.5)
        A1[i+j*M, (i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
        A2[i+j*M, i+(j-1)*M] += dt*(-kappa*(theta-V[j])/(JV[j]*hl))
        A2[i+j*M, i+j*M] += dt*(-r*0.5)
        A2[i+j*M, i+(j)*M] += dt*(kappa*(theta-V[j])/(JV[j]*hl))
    for j in range(1, L-1):
        # boundary conditions i=0,M-1.
        iBarrier = 1
        if B == 0:
            i = 0
            A1[i+j*M, (i+1)+j*M] += dt*((r-q)/(hm))
            A1[i+j*M, i+j*M] += dt*(-r*0.5)
            A1[i+j*M, (i)+j*M] += dt*(-(r-q)/(hm))
            A2[i+j*M, i+j*M] += dt*(-r*0.5)
        else:
            iBarrier = iBarrierList[j]  # S[i-1]<B<=S[i]
            BC[iBarrier-1+j*M, iBarrier-1+j *M] = (S[iBarrier+j*M]-B)/(S[iBarrier+j*M]-S[iBarrier-1+j*M])-1
            BC[iBarrier-1+j*M, iBarrier+j *M] = (B-S[iBarrier-1+j*M])/(S[iBarrier+j*M]-S[iBarrier-1+j*M])
        i = M-1
        A1[i+j*M, (i-1)+j*M] += dt*(-(r-q-0.5*V[j]+kappa*(theta-V[j])*alpha))/(hm)
        A1[i+j*M, i+j*M] += dt*(-r*0.5)
        A1[i+j*M, (i)+j*M] += dt*((r-q)-0.5*V[j]+kappa*(theta-V[j])*alpha)/(hm)
        A2[i+j*M, i+j*M] += dt*(-r*0.5)
        for i in range(iBarrier, M-1):
            svj=sigma*sigma*V[j]/(JV[j])
            driftj=kappa*(theta-V[j])
            svi=V[j]*(1+sigma*sigma*alpha*alpha+2*rho*sigma*alpha)
            drifti=(r-q-0.5*V[j]+kappa*(theta-V[j])*alpha)
            if useExponentialFitting:
                if abs(drifti*hm/svi) > upwindingThreshold:
                    svi=drifti*hm/math.tanh(drifti*hm/svi)
                if driftj != 0 and abs(driftj*hl/svj) > upwindingThreshold:
                    svj=driftj*hl/math.tanh(driftj*hl/svj)
            A1[i+j*M, (i+1)+j*M] += dt*(svi*0.5/(hm*hm)+drifti/(2*hm))
            A1[i+j*M, i+j*M] += dt*(-(svi)/(hm*hm)-r*0.5)
            A1[i+j*M, (i-1)+j*M] += dt*(svi*0.5/(hm*hm)-drifti/(2*hm))
            A2[i+j*M, i+(j+1)*M] += dt * (0.5*svj/(JVm[j+1]*hl*hl)+driftj/(2*JV[j]*hl))
            A2[i+j*M, i+j*M] += dt * (-r*0.5-0.5*svj/(hl*hl)*(1.0/JVm[j+1]+1.0/JVm[j]))
            A2[i+j*M, i+(j-1)*M] += dt *(0.5*svj/(JVm[j]*hl*hl)-driftj/(2*JV[j]*hl))
    A1tri, A2tri, indices, indicesInv=createTridiagonalIndices(M, L)
    A0=0.0
    return F, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv


def createTridiagonalIndices(M, L):
    A1tri=np.zeros((3, M*L))
    A2tri=np.zeros((3, M*L))
    indices=np.zeros(M*L, dtype=int)
    indicesInv=np.zeros(M*L, dtype=int)
    for i in range(M):
        for j in range(L):
            indices[i+j*M]=j+i*L
            indicesInv[j+i*L]=i+j*M
    return A1tri, A2tri, indices, indicesInv


def priceCallTransformed(method, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L):
    isCall=False
    damping="Euler"  # "None" "One", "Euler"
    useVLinear=False
    useExponentialFitting=True
    alpha=-rho/sigma
    epsilon=1e-3
    dChi=4*kappa*theta/(sigma*sigma)
    chiN=4*kappa*math.exp(-kappa*T)/(sigma*sigma*(1-math.exp(-kappa*T)))
    vmax=ncx2.ppf((1-epsilon), dChi, v0*chiN)*math.exp(-kappa*T)/chiN
    vmin=ncx2.ppf((epsilon), dChi, v0*chiN)*math.exp(-kappa*T)/chiN
    # print("vmax",vmin,vmax, 10*v0)
    vmin=max(1e-3, vmin)  # Peclet explodes at V=0!
    vmin=0.0
    # vmax=10*v0#0.28
    V=np.linspace(vmin, vmax, L)
    hl=V[1]-V[0]
    JV=np.ones(L)
    JVm=np.ones(L)
    if not useVLinear:
        vscale=v0*2
        # 1e-4,math.sqrt(vmax),L)  #ideally, concentrated around v0: V=sinh((w-w0)/c). w unif
        u=np.linspace(0, 1, L)
        c1=math.asinh((vmin-v0)/vscale)
        c2=math.asinh((vmax-v0)/vscale)
        V=v0 + vscale*np.sinh((c2-c1)*u+c1)
        hl=u[1]-u[0]
        JV=vscale*(c2-c1) * np.cosh((c2-c1)*u+c1)
        JVm=vscale*(c2-c1) * np.cosh((c2-c1)*(u-hl/2)+c1)
    # max(4*math.sqrt(theta*T),(0.5*math.sqrt(v0*T)+abs(alpha*vmax)))
    Xspan=4*math.sqrt(theta*T)
    Kinv=cFunc.solve(K)
    Xmin=math.log(Kinv)-Xspan+alpha*vmin #rho <0, alpha > 0
    Xmax=math.log(Kinv)+Xspan+alpha*vmax
    # print("Xmin",Xmin,"Xmax",Xmax)
    X=np.linspace(Xmin, Xmax, M)
    hm=X[1]-X[0]
    # V
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
    F0=np.zeros(M*L)
    S=np.zeros(M*L)
    lnK=math.log(K)
    sign=1
    if not isCall:
        sign=-1
    for j in range(L):
        for i in range(M):
            S[i+j*M]=np.exp(X[i]-alpha*V[j])
    Sc=np.array([cFunc.evaluate(T, Si) for Si in S])
    F0=np.maximum(sign*(Sc-K), 0)
    iBarrierList=np.zeros(L, dtype='int')
    for j in range(L):
        iBarrierList[j]=np.searchsorted(Sc[j*M:(j+1)*M], B)  # S[i-1]<B<=S[i]
    # print("iBarrierList",iBarrierList)
    F, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv=createSystemTrans(
        useExponentialFitting, B, iBarrierList, Sc, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L)
    A1=A1.tocsc()
    A2=A2.tocsc()
    A1tri[1, :]=A1.diagonal(k=0)
    A1tri[-1, :-1]=A1.diagonal(k=-1)
    A1tri[0, 1:]=A1.diagonal(k=1)
    A2tri[1, :]=A2.diagonal(k=0)[indicesInv]
    A2i=A2[:, indicesInv]
    A2i=A2i[indicesInv, :]
    A2tri[-1, :-1]=A2i.diagonal(k=-1)
    A2tri[0, 1:]=A2i.diagonal(k=1)
    I=identity(M*L, format="csc")
    jv0=np.searchsorted(V, v0)
    PayoffTime=np.zeros((N, M))
    start=time.time()
    if damping == "Euler":
        A=A0+A1+A2
        a=0.5
        Li=I+a*A+BC
        lu=sla.splu(Li)
        updatePayoffBoundaryTrans(F, Sc, B, iBarrierList, M, L)
        F=lu.solve(F)
        updatePayoffBoundaryTrans(F, Sc, B, iBarrierList, M, L)
        F=lu.solve(F)
        N -= 1
        # updatePayoffExplicitTrans(F, Sc, B, iBarrierList, M, L)
        PayoffTime[N][:]=F[jv0*M:jv0*M+M]
    if method == "EU":
        A=A0+A1+A2
        Li=I+A+BC
        lu=sla.splu(Li)
        for i in range(N):
            updatePayoffBoundaryTrans(F, Sc, B, iBarrierList, M, L)
            F=lu.solve(F)
    elif method == "CS":
        a=0.5
        lu1=sla.splu(I+a*A1+BC)
        lu2=sla.splu(I+a*A2+BC)
        for i in range(N):
            # updatePayoffExplicitTrans(F,S,B,iBarrierList,M,L)
            Y0=(I-A0-A1-A2)*F  # explicit
            # updatePayoffExplicitTrans(Y0,S,B,iBarrierList,M,L)
            Y0r=Y0+a*A1*F
            updatePayoffBoundaryTrans(Y0r, Sc, B, iBarrierList, M, L)
            Y1=lu1.solve(Y0r)
            Y1r=Y1+a*A2*F
            updatePayoffBoundaryTrans(Y1r, Sc, B, iBarrierList, M, L)
            Y2=lu2.solve(Y1r)
            Y0t=Y0 - 0.5*(A0*Y2-A0*F)
            Y0r=Y0t + a*A1*F
            updatePayoffBoundaryTrans(Y0r, Sc, B, iBarrierList, M, L)
            Y1t=lu1.solve(Y0r)
            Y1r=Y1t+a*A2*F
            updatePayoffBoundaryTrans(Y1r, Sc, B, iBarrierList, M, L)
            Y2t=lu2.solve(Y1r)
            F=Y2t
    elif method == "DO":
        a=0.5
        lu1=sla.splu(I+a*A1+BC)
        lu2=sla.splu(I+a*A2+BC)
        for i in range(N):
            updatePayoffExplicitTrans(Y0r, Sc, B, iBarrierList, M, L)
            Y0=F-(A0+A1+A2)*F  # explicit
            Y0r=Y0+a*(A1*F)
            updatePayoffBoundaryTrans(Y0r, Sc, B, iBarrierList, M, L)
            Y1=lu1.solve(Y0r)
            Y1r=Y1+a*(A2*F)
            updatePayoffBoundaryTrans(Y1r, Sc, B, iBarrierList, M, L)
            Y2=lu2.solve(Y1r)
            F=Y2
    elif method == "PR":  # peaceman-rachford strikwerda
        a=0.5
        ti=T
        dt=1.0/N
        if B == 0:
            A1tri *= a
            A1tri[1, :] += 1
            A2tri *= a
            A2tri[1, :] += 1
            for i in range(N):
                # updatePayoffExplicitTrans(F,S,B,iBarrierList,M,L)
                Y0=F-a*(A2*F)
                Y1=la.solve_banded(
                    (1, 1), A1tri, Y0, overwrite_ab=False, overwrite_b=True, check_finite=False)
                Y1t=Y1-a*(A1*Y1)
                Y1t=Y1t[indicesInv]
                Y2t=la.solve_banded(
                    (1, 1), A2tri, Y1t, overwrite_ab=False, overwrite_b=True, check_finite=False)
                F=Y2t[indices]
        else:
            # updatePayoffExplicitTrans(F, Sc, B, iBarrierList, M, L)
            for i in range(N):
                ti -= dt*0.5
                Sc=np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j]=np.searchsorted(
                        Sc[j*M:(j+1)*M], B)  # S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv=createSystemTrans(
                    useExponentialFitting, B, iBarrierList, Sc, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L)
                # updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L)
                Y0=(I)*F-a*(A2*F)
                updatePayoffBoundaryTrans(Y0, Sc, B, iBarrierList, M, L)
                lu1=sla.splu(I+a*A1+BC)
                Y1=lu1.solve(Y0)
                Y1t=(I)*Y1-a*(A1*Y1)
                # boundary may not be adequate for A2 direction.
                updatePayoffBoundaryTrans(Y1t, Sc, B, iBarrierList, M, L)
                lu2=sla.splu(I+a*A2+BC)
                Y2=lu2.solve(Y1t)
                F=Y2
                ti -= dt*0.5
                # updatePayoffExplicitTrans(F, Sc, B, iBarrierList, M, L)
                PayoffTime[N-i][:]=F[jv0*M:jv0*M+M]
    elif method == "CN":
        a=0.5
        if B == 0:
            A=A0+A1+A2
            Li=I+a*A+BC
            Le=I-(1-a)*A
            lu=sla.splu(Li)
            for i in range(N):
                # updatePayoffExplicitTrans(F,Sc,B,iBarrierList,M,L)
                Y0=Le*F
                updatePayoffBoundaryTrans(Y0, Sc, B, iBarrierList, M, L)
                F=lu.solve(Y0)
        else:
            for i in range(N):
                ti=T*(N-i)/N
                ti=ti-a*T/N
                Sc=np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j]=np.searchsorted(
                        Sc[j*M:(j+1)*M], B)  # S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv=createSystemTrans(
                    useExponentialFitting, B, iBarrierList, Sc, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L)
                A=A0+A1+A2
                Li=I+a*A+BC
                Le=I-(1-a)*A
                lu=sla.splu(Li)
                Y0=Le*F
                updatePayoffBoundaryTrans(Y0, Sc, B, iBarrierList, M, L)
                F=lu.solve(Y0)
                updatePayoffExplicitTrans(F, Sc, B, iBarrierList, M, L)
                ti=ti-a*T/N
    else:  # if method =="LS":
        a=1 - math.sqrt(2)/2
        dt=-T/N
        if B == 0:
            A=A0+A1+A2
            Li=I+a*A+BC
            lu=sla.splu(Li)  # ilu(Li,drop_tol=1e-10,fill_factor=1000)
            for i in range(N):
                updatePayoffBoundaryTrans(F, Sc, B, iBarrierList, M, L)
                F1=lu.solve(F)
                updatePayoffBoundaryTrans(F1, Sc, B, iBarrierList, M, L)
                F2=lu.solve(F1)
                F=(1+math.sqrt(2))*F2 - math.sqrt(2)*F1
        else:
            for i in range(N):
                ti=T*(N-i)/N
                ti=ti+a*dt
                Sc=np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j]=np.searchsorted(
                        Sc[j*M:(j+1)*M], B)  # S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv=createSystemTrans(
                    useExponentialFitting, B, iBarrierList, Sc, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L)
                updatePayoffBoundaryTrans(F, Sc, B, iBarrierList, M, L)
                A=A0+A1+A2
                Li=I+a*A+BC
                lu=sla.splu(Li)  # ilu(Li,drop_tol=1e-10,fill_factor=1000)
                F1=lu.solve(F)
                ti=ti+a*dt
                Sc=np.array([cFunc.evaluate(ti, Si) for Si in S])
                for j in range(L):
                    iBarrierList[j]=np.searchsorted(
                        Sc[j*M:(j+1)*M], B)  # S[i-1]<B<=S[i]
                Ftemp, A0, A1, A2, BC, A1tri, A2tri, indices, indicesInv=createSystemTrans(
                    useExponentialFitting, B, iBarrierList, Sc, F0, V, JV, JVm, r, q, kappa, theta, rho, sigma, alpha, hm, hl, T, N, M, L)
                updatePayoffBoundaryTrans(F1, Sc, B, iBarrierList, M, L)
                A=A0+A1+A2
                Li=I+a*A+BC
                lu=sla.splu(Li)  # ilu(Li,drop_tol=1e-10,fill_factor=1000)
                F2=lu.solve(F1)
                F=(1+math.sqrt(2))*F2 - math.sqrt(2)*F1
    end=time.time()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # Tarr = np.arange(0, T, T/(N+1))
    # Xmesh, Tmesh = np.meshgrid(X,Tarr)
    # print("T",Tmesh.shape)
    # print("X", Xmesh.shape)
    # print("P", PayoffTime.shape)
    # surf = ax.plot_surface(Tmesh, Xmesh, np.minimum(np.maximum(PayoffTime,-0.05),0.1), cmap=cm.coolwarm, linewidth=0, antialiased=True)
    # #ax.set_zlim(-0.05,0.05)
    # plt.show()
    Payoff=F.reshape(L, M)
    # print("Payoff V=0",Payoff[0])
    #
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
    Payoffi=interpolate.RectBivariateSpline(V, X, Payoff, kx=3, ky=3, s=0)
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
    maxError=0.0
    for spot, refPrice in zip(spotArray, priceArray):
        x0=math.log(spot)+alpha*v0
        price=Payoffi(v0, x0)[0][0]
        delta=Payoffi(v0, x0, dy=1)[0][0]
        gamma=Payoffi(v0, x0, dy=2)[0][0]
        error=price - refPrice
        if abs(error) > maxError:
            maxError=abs(error)
        if B == 0:
            print(spot, method, N, M, L, price, delta, gamma, error, end-start)
    if B == 0:
        pass  # print(method,N,M,L,maxError,end-start)
    else:
        x0=math.log(K)+alpha*v0
        print(method, N, M, L, Payoffi(v0, x0)[0][0], end-start)


class IdentityFunction:
    def __init__(self):
        pass
    def evaluate(self, t, z):
        return z
    def evaluateSlice(self, z):
        return z
    def solve(self, strike):
        return strike

class CollocationFunction:
    X=[]
    A=[]
    B=[]
    C=[]
    leftSlope=0.0
    rightSlope=0.0
    T=0.0
    def __init__(self, X, A, B, C, leftSlope, rightSlope, T):
        self.X=X
        self.A=A
        self.B=B
        self.C=C
        self.leftSlope=leftSlope
        self.rightSlope=rightSlope
        self.T=T
    def evaluateSlice(self, z):
        if z <= self.X[0]:
            return self.leftSlope*(z-self.X[0]) + self.A[0]
        elif z >= self.X[-1]:
            return self.rightSlope*(z-self.X[-1])+self.A[-1]
        i=np.searchsorted(self.X, z)  # x[i-1]<z<=x[i]
        if i > 0:
            i -= 1
        h=z-self.X[i]
        return self.A[i] + h*(self.B[i]+h*self.C[i])
    def evaluate(self, t, z):
        # linear interpolation between slice at t=0 and slice T.
        return t/self.T * self.evaluateSlice(z) + (1.0-t/self.T)*z
    def solve(self, strike):
        if strike < self.A[0]:
            sn=self.leftSlope
            return (strike-self.A[0])/sn + self.X[0]
        elif strike > self.A[-1]:
            sn=self.rightSlope
            return (strike-self.A[-1])/sn + self.X[-1]
        i=np.searchsorted(self.A, strike)  # a[i-1]<strike<=a[i]
        # print("index",self.A[i-1],strike,self.A[i],len(self.A))
        if abs(self.A[i]-strike) < 1e-10:
            return self.X[i]
        if abs(self.A[i-1]-strike) < 1e-10:
            return self.X[i-1]
        if i == 0:
            i += 1
        x0=self.X[i-1]
        c=self.C[i-1]
        b=self.B[i-1]
        a=self.A[i-1]
        d=0
        cc=a + x0*(-b+x0*(c-d*x0)) - strike
        bb=b + x0*(-2*c+x0*3*d)
        aa=-3*d*x0 + c
        allck=np.roots([aa, bb, cc])
        for ck in allck:
            if abs(ck.imag) < 1e-10 and ck.real >= self.X[i-1]-1e-10 and ck.real <= self.X[i]+1e-10:
                return ck.real
        raise Exception("no roots found in range", allck, strike,
                        aa, bb, cc, i, self.X[i-1], self.X[i])


def priceSX5ETime(method):
    # Spline 1e-3 penalty
    A=[0.6266758553145932, 0.8838690008217314, 0.9511741483703275, 0.9972169412308787,
         1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
    B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555,
         1.1476195823811128, 1.173600641673776, 1.1472056638621118, 0.918270335988941]
    C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754,
         0.31401193651971954, -0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
    X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388,
         1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
    leftSlope=0.8329310535215612
    rightSlope=0.2668764075068484
    kappa=0.35
    theta=0.321
    sigma=1.388
    rho=-0.63
    r=0.0
    q=0.0
    v0=0.133
    T=0.4986301369863014
    cFunc = IdentityFunction()
    #cFunc=CollocationFunction(X, A, B, C, leftSlope, rightSlope, T)
    K=1.0
    spotArray=[1.0]  # max(s-K) = max(s/K-1)*K
    priceArray=[0.07278065]
    M=256  # X
    L=32  # V
    B=0.8
    # Ns = [4096,2048,1024, 512, 256, 128, 64, 32, 16, 8 ,4] #timesteps
    Ns=[4096, 1024, 768, 512, 384, 256, 192, 128, 96,
          64, 56, 48, 32, 24, 16, 12, 8, 6, 4]  # timesteps
    # Ns = [72,60,12]
    Ns.reverse()
    for N in Ns:
        priceCallTransformed(method, spotArray, priceArray, v0,
                             kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L)


def main():
    method="CN"  # "LODLS" "CS" "LS","CN","DO"
    priceSX5ETime(method)

A=[0.6266758553145932, 0.8838690008217314, 0.9511741483703275, 0.9972169412308787,
    1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
B=[0.8329310535215612, 0.5486175716699259, 1.0783076034285555,
    1.1476195823811128, 1.173600641673776, 1.1472056638621118, 0.918270335988941]
C=[-0.38180731761048253, 3.2009663415588276, 0.8377175268235754,
    0.31401193651971954, -0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
X=[0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388,
    1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
for i in range(len(A)):
    prinln(X[i], " & ", A[i], " & ", B[i], " & ", C[i], "\\\\")

if __name__ == '__main__':
    main()
