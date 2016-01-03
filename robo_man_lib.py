import numpy as np
import sympy.mpmath as symath
symath.mp.dps = 15
symath.mp.pretty = True
from math import sqrt, sin, cos, tan, acos, pi
import matplotlib.pyplot as plt

def matmult(*x):
    return reduce(np.dot, x)

def Sth(S,th):
    S = [x * th for x in S]
    return S

def CheckRotMat(R):
    if R.shape[0] is 3 and R.shape[1] is 3:
        if np.max(matmult(R.T,R)-np.eye(3)) < 1e-3:
            if np.linalg.det(R) - 1 < 1e-3:
                return True
    return False

def RotInv(R):
    if CheckRotMat(R): return R.T
    else:
        return "Not a rotation matrix!"

def VecToso3(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
        ])

def so3ToVec(w):    
    return np.array([-w[1,2], w[0,2], -w[0,1]])

def AxisAng3(r):
    return r/np.linalg.norm(r), np.linalg.norm(r)

def MatrixExp3(r):
    if np.max(np.absolute(r)) < 1e-12:
        return np.eye(3)
    else:
        th = AxisAng3(r)[1]
        w = VecToso3(AxisAng3(r)[0])
        return np.eye(3) + (sin(th))*w + (1-cos(th))*matmult(w,w)

def MatrixLog3(R):
    if CheckRotMat(R):
        if np.max(R-np.eye(3)) < 1e-3:
            return 0
        elif np.absolute(np.trace(R) + 1) < 1e-5:
            if R[2,2] != -1.: 
                return pi*(1/sqrt(2*(1 + R[2,2]))) * np.array([R[0,2], R[1,2], 1+R[2,2]])
            elif R[1,1] != -1.: 
                return pi*(1/sqrt(2*(1 + R[1,1]))) * np.array([R[0,1], 1+R[1,1], R[2,1]])
            elif R[0,0] != -1.: 
                return pi*(1/sqrt(2*(1 + R[0,0]))) * np.array([1+R[0,0], R[1,0], R[2,0]])
        else:
            th = acos((np.trace(R) - 1)/2.)
            w = (1/(2*sin(th))) * (R - R.T)            
            return th*so3ToVec(w)
    else:
        return "Not a rotation matrix!" 

def RpToTrans(R, p):
    p.shape = (np.size(p),1)
    if CheckRotMat(R):
        temp = np.zeros((1,3))        
        return np.hstack((np.vstack((R,temp)),np.vstack((p,1))))
    else:
        return "Not a rotation matrix!"

def TransToRp(T):
    return T[0:3,0:3], T[0:3,3]

def TransInv(T):
    if CheckRotMat(TransToRp(T)[0]):
        T[0:3,0:3] = np.transpose(TransToRp(T)[0])
        T[0:3,3] = matmult(-T[0:3,0:3],TransToRp(T)[1])
        return T
    else:
        return "No rotation matrix found!"

def VecTose3(V):
    V.shape = (np.size(V),1)
    w = VecToso3(V[0:3,0])
    v = V[3:6,0]
    v.shape = (np.size(v),1)
    temp = np.zeros((1,4))
    return np.vstack((np.hstack((w,v)),temp))

def se3ToVec(V):
    return np.array([-V[1,2], V[0,2], -V[0,1], V[0,3], V[1,3], V[2,3]])

def Adjoint(T):
    R, p = TransToRp(T)
    if CheckRotMat(R):    
        temp = np.zeros((3,3))
        return np.vstack((np.hstack((R,temp)),np.hstack((matmult(VecToso3(p),R),R))))
    else:
        return "No rotation matrix!"

def ScrewToAxis(q, s_hat, h):
    q, s_hat = np.array(q), np.array(s_hat)
    if np.absolute(np.linalg.norm(s_hat))-1 < 1e-3:
        cp = np.cross(-s_hat,q)
        cp.shape = (np.size(cp), 1)
        s_hat.shape = (np.size(s_hat), 1)
        return np.vstack((s_hat, cp + (h*s_hat)))
    else:
        return "s_hat is not a unit vector!"

def AxisAng6(S):
    S = np.array(S)
    S.shape = (np.size(S),1)
    if np.max(np.absolute(S[:,0])) < 1e-12:
        return S, 0
    elif np.max(np.absolute(S[0:3,0])) < 1e-12:
        th = np.linalg.norm(S[3:6,0])
        S[3:6,0] = S[3:6,0]/th
        return S, th    
    else:
        return S/np.linalg.norm(S[0:3,0]), np.linalg.norm(S[0:3,0])

def MatrixExp6(S): 
    S = np.array(S) 
    S.shape = (np.size(S),1)
    r = np.array(S[0:3,0])    
    S, th = AxisAng6(S)
    w = VecToso3(S[0:3,0])
    v = np.array(S[3:6,0])    
    G_v = matmult((th*np.eye(3) + (1-cos(th))*w + ((th-sin(th))*matmult(w,w))),v)
    G_v.shape = (np.size(G_v),1)
    return np.hstack((np.vstack((MatrixExp3(r), np.zeros((1,3)))), np.vstack((G_v, 1))))

def MatrixLog6(T):
    R, p = TransToRp(T)
    p.shape = (np.size(p),1)
    if CheckRotMat(R):
        if np.max(R-np.eye(3)) < 1e-3:
            return np.vstack((np.zeros((3,1)), p))
        elif np.absolute(np.trace(R) + 1) < 1e-3:   
            th = pi
            w = VecToso3(MatrixLog3(R))
            G_inv = (1./th)*np.eye(3) - (1./2)*w + ((1/th)-((1./2)*(1/tan(th/2))))*matmult(w,w)
            v = th*matmult(G_inv, p)
            v.shape = (np.size(v),1)
            w = th*so3ToVec(w)
            w.shape = (np.size(w),1)
            return np.vstack((w,v))
        else:
            th = acos((np.trace(R) - 1)/2.)
            w = (1./(2*sin(th))) * (R - RotInv(R)) 
            G_inv = (1./th)*np.eye(3) - (1./2)*w + ((1/th)-((1./2)*(1/tan(th/2))))*matmult(w,w)
            v = th*matmult(G_inv, p)
            v.shape = (np.size(v),1)
            w = th*so3ToVec(w)
            w.shape = (np.size(w),1)
            return np.vstack((w,v))
    else:
        return "No rotation matrix!"

def FKinFixed(M, S_list, th_list): 
    M = np.reshape(M, (4,4))
    if CheckRotMat(TransToRp(M)[0]):
        n = len(S_list)        
        T = M        
        for i in range(n-1,-1,-1):
            S_i = np.array(S_list[i])
            S_i.shape = (np.size(S_i),1)
            th_i = th_list[i]
            T = matmult(MatrixExp6(S_i*th_i), T)        
        return T
    else:
        return "No rotation matrix found!"

def FKinBody(M, B_list, th_list):
    M = np.reshape(M, (4,4))
    if CheckRotMat(TransToRp(M)[0]):
        n = len(B_list)
        T = M
        for i in range(n):
            B_i = np.array(B_list[i])
            B_i.shape = (np.size(B_i),1)
            th_i = th_list[i]
            T = matmult(T,MatrixExp6(B_i*th_i))        
        return T
    else:
        return "No rotation matrix found!"
    
def FixedJacobian(S_list, th_list):
    n = len(S_list)
    J = np.zeros((6,n))    
    J[:,0] = np.array(S_list[0])
    T = np.eye(4)
    for i in range(1,n):
        S_i = np.array(S_list[i-1])
        S_i.shape = (np.size(S_i),1) 
        th = th_list[i-1]            
        S = np.array(S_list[i])
        S.shape = (np.size(S),1)
        T_i = MatrixExp6(S_i*th)
        T = matmult(T, T_i)
        V = matmult(Adjoint(T),S)
        J[:,i] = V[:,0]   
    return J 

def BodyJacobian(B_list, th_list):
    n = len(B_list)
    J = np.zeros((6,n))    
    J[:,n-1] = np.array(B_list[n-1])
    T = np.eye(4)
    for i in reversed(range(0,n-1)):
        B_i = np.array(B_list[i+1])
        B_i.shape = (np.size(B_i),1) 
        th = th_list[i+1]            
        B = np.array(B_list[i])
        B.shape = (np.size(B),1)
        T_i = MatrixExp6(B_i*-th)
        T = matmult(T, T_i)
        V = matmult(Adjoint(T),B)
        J[:,i] = V[:,0]   
    return J

def IKinBody(B_list, M, T_sd, th_list, e_w, e_v):
    T_sd = np.reshape(T_sd, (4,4))
    th = np.array(th_list)
    th.shape = (np.size(th),1)
    TH = th.T
    T_sb = FKinBody(M, B_list, th)
    V_b = MatrixLog6(matmult(TransInv(T_sb),T_sd))    
    w_b = np.linalg.norm(V_b[0:3,0])
    v_b = np.linalg.norm(V_b[3:6,0])
    
    i = 0
    max_its = 100
    while i < max_its and (w_b > e_w or v_b > e_v):        
        th = th + matmult(np.linalg.pinv(BodyJacobian(B_list, th)),V_b)      
        TH = np.vstack((TH, th.T))
        T_sb = FKinBody(M, B_list, th)        
        V_b = MatrixLog6(matmult(TransInv(T_sb),T_sd))  
        w_b = np.linalg.norm(V_b[0:3,0])
        v_b = np.linalg.norm(V_b[3:6,0])   
        i += 1        
    return TH   

def IKinFixed(S_list, M, T_sd, th_list, e_w, e_v):
    T_sd = np.reshape(T_sd, (4,4))
    th = np.array(th_list)
    th.shape = (np.size(th),1)
    TH = th.T
    T_sb = FKinFixed(M, S_list, th)
    V_b = MatrixLog6(matmult(TransInv(T_sb),T_sd))
    V_s = matmult(Adjoint(T_sb),V_b)
    w_s = np.linalg.norm(V_s[0:3,0])
    v_s = np.linalg.norm(V_s[3:6,0])
    
    i = 0
    max_its = 100
    while i < max_its and (w_s > e_w or v_s > e_v):
        th = th + matmult(np.linalg.pinv(FixedJacobian(S_list, th)),V_s)
        TH = np.vstack((TH, th.T))
        T_sb = FKinFixed(M, S_list, th)
        V_b = MatrixLog6(matmult(TransInv(T_sb),T_sd))
        V_s = matmult(Adjoint(T_sb),V_b)
        w_s = np.linalg.norm(V_s[0:3,0])
        v_s = np.linalg.norm(V_s[3:6,0])        
        i += 1        
    return TH 

def CubicTimeScaling(t, T):
    return 3*(t/T)**2 - 2*(t/T)**3

def QuinticTimeScaling(t, T):
    return 10*(t/T)**3 - 15*(t/T)**4 + 6*(t/T)**5

def JointTrajectory(thst, then, T, N, ts):
    try:
        thst, then = np.array(thst), np.array(then)
        thst.shape, then.shape = (1,np.size(thst)), (1,np.size(then))
        TH = thst
        t0 = T/(N-1)
        t = 0    
        for i in range(2, N+1):
            t += t0
            if ts == 3:
                s = CubicTimeScaling(t, T)
            elif ts == 5:
                s = QuinticTimeScaling(t, T)
            ths = thst + s*(then - thst)
            TH = np.vstack((TH, ths))                    
        return TH
    except:
        print "Please check input parameters!"
    
def ScrewTrajectory(Xst, Xen, T, N, ts):
    try:
        Xst, Xen = np.reshape(Xst, (4,4)), np.reshape(Xen, (4,4))
        Xlist = [Xst]    
        t0 = T/(N-1)
        t = 0    
        for i in range(2, N+1):
            t += t0
            if ts == 3:
                s = CubicTimeScaling(t, T)
            elif ts == 5:
                s = QuinticTimeScaling(t, T)
            Xs = np.dot(Xst,MatrixExp6(s*MatrixLog6(np.dot(TransInv(Xst),Xen))))
            Xlist.append(Xs)               
        return Xlist
    except:
        print "Please check input parameters!"
        
def CartesianTrajectory(Xst, Xen, T, N, ts):
    try:
        Xst, Xen = np.reshape(Xst, (4,4)), np.reshape(Xen, (4,4))
        Rst, pst = TransToRp(Xst)
        Ren, pen = TransToRp(Xen)
        pst.shape, pen.shape = (np.size(pst),1), (np.size(pen),1)
        Xlist = [Xst]
        t0 = T/(N-1)
        t = 0    
        for i in range(2, N+1):
            t += t0
            if ts == 3:
                s = CubicTimeScaling(t, T)
            elif ts == 5:
                s = QuinticTimeScaling(t, T)
            ps = pst + s*(pen-pst)
            Rs = np.dot(Rst,MatrixExp3(s*MatrixLog3(np.dot(Rst.T,Ren))))
            Xs = RpToTrans(Rs, ps)
            Xlist.append(Xs)               
        return Xlist
    except:
        print "Please check input parameters!"
        
def RoboTrajectory(thst, then, T, dt, ts):
    try:
        thst, then = np.array(thst), np.array(then)
        thst.shape, then.shape = (1,np.size(thst)), (1,np.size(then))
        n = thst.shape[1]
        N = int(T/dt) 
        robotraj = []
        thlistset = JointTrajectory(thst, then, T, N+1, ts)
        robotraj.append(thlistset)
        thdotlistlist = []
        thdotdotlistlist = []
        ti = 0    
        for i in range(0, N+1):            
            if ts == 3:
                thdotlist = symath.diff(lambda t: thst + CubicTimeScaling(t,T) * (then-thst), ti)
                thdotdotlist = symath.diff(lambda t: thst + CubicTimeScaling(t,T) * (then-thst), ti, 2)
            elif ts == 5:
                thdotlist = symath.diff(lambda t: thst + QuinticTimeScaling(t,T) * (then-thst), ti)
                thdotdotlist = symath.diff(lambda t: thst + QuinticTimeScaling(t,T) * (then-thst), ti, 2)
            ti += dt
            thdotlistlist.append(thdotlist.astype(np.float64))
            thdotdotlistlist.append(thdotdotlist.astype(np.float64))
        thdotlistset = thdotlistlist[0]
        thdotdotlistset = thdotdotlistlist[0]
        for i in range(1,N+1):
            thdotlistset = np.vstack((thdotlistset, thdotlistlist[i]))
            thdotdotlistset = np.vstack((thdotdotlistset, thdotdotlistlist[i]))        
        robotraj.append(thdotlistset)
        robotraj.append(thdotdotlistset)
        return robotraj
    except:
        print "Please check input parameters!"

def adV(V):
    V = np.array(V)
    V.shape = (np.size(V),1)
    return np.vstack((np.hstack((VecToso3(V[0:3,0]),np.zeros((3,3)))),np.hstack((VecToso3(V[3:6,0]),VecToso3(V[0:3,0])))))

def LieBracket(V1, V2):
    V1, V2 = np.array(V1), np.array(V2)
    V1.shape, V2.shape = (np.size(V1),1), (np.size(V2),1)    
    return matmult(adV(V1), V2)

def InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist):    
    gvec, Ftip = np.array(gvec), np.array(Ftip)
    gvec.shape, Ftip.shape = (np.size(gvec),1), (np.size(Ftip),1)
    for i in range(len(Mlist)):
        Mlist[i] = np.reshape(Mlist[i], (4,4))  
    for i in range(len(Glist)):
        Glist[i] = np.reshape(Glist[i], (6,6))
    for i in range(len(Slist)):
        Slist[i] = np.reshape(Slist[i], (6,1))
    T = []
    V = []
    Vdot = []
    M = []
    A = []
    temp = np.zeros((6,1))
    V.append(temp)
    Vdot.append(np.vstack((np.zeros((3,1)), gvec)))
    M.append(np.eye(4,4))
    n = len(thlist)     
    for i in range(1,n+1):
        Si = np.copy(Slist)[i-1]
        M.append(matmult(np.copy(M)[i-1],np.copy(Mlist)[i-1]))
        A.append(matmult(Adjoint(TransInv(np.copy(M)[i])), Si))
        T.append(matmult(np.copy(Mlist)[i-1], MatrixExp6(Sth(np.copy(A)[i-1],np.copy(thlist)[i-1]))))
        V.append(matmult(Adjoint(TransInv(np.copy(T)[i-1])),np.copy(V)[i-1]) + np.copy(A)[i-1]*np.copy(thdotlist)[i-1])
        Vdot.append(matmult(Adjoint(TransInv(np.copy(T)[i-1])), np.copy(Vdot)[i-1]) + LieBracket(np.copy(V)[i],np.copy(A)[i-1])*np.copy(thdotlist)[i-1] + np.copy(A)[i-1]*np.copy(thdotdotlist)[i-1])
        
    tau = []
    T.append(Mlist[n])
    F = []
    F.append(Ftip)    
    for i in reversed(range(1,n+1)):
        F.insert(0,matmult((Adjoint(TransInv(np.copy(T)[i]))).T,np.copy(F)[0]) + matmult(np.copy(Glist)[i-1],np.copy(Vdot)[i]) - matmult((adV(np.copy(V)[i])).T,matmult(np.copy(Glist)[i-1],np.copy(V)[i])))
        tau.insert(0,matmult((np.copy(F)[0]).T,np.copy(A)[i-1]))
    tau = np.array(tau)
    tau.shape = (np.size(tau),1)
    
    return tau    

def InertiaMatrix(thlist, Mlist, Glist, Slist):
    n = len(thlist)
    gvec = [0,0,0]
    Ftip = [0,0,0,0,0,0]
    Mth = np.array([])
    for i in range(n):
        thdotlist = []
        thdotdotlist = []
        for j in range(n):
            thdotlist.append(0)
            if i == j:
                thdotdotlist.append(1)
            else:
                thdotdotlist.append(0)
        Mi = InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist)
        Mth = np.append(Mth, Mi)
    return np.reshape(Mth, (n,n)).T

def CoriolisForces(thlist, thdotlist, Mlist, Glist, Slist):
    n = len(thlist) 
    thdotdotlist = []
    for i in range(n):
        thdotdotlist.append(0)
    gvec = [0,0,0]
    Ftip = [0,0,0,0,0,0]
    return InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist)

def GravityForces(thlist, gvec, Mlist, Glist, Slist):
    n = len(thlist)
    thdotlist = []
    thdotdotlist = []
    for i in range(n):
        thdotlist.append(0)
        thdotdotlist.append(0)
    Ftip = [0,0,0,0,0,0]
    return InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist)

def EndEffectorForces(thlist, Ftip, Mlist, Glist, Slist):
    n = len(thlist)
    thdotlist = []
    thdotdotlist = []
    for i in range(n):
        thdotlist.append(0)
        thdotdotlist.append(0)
    gvec = [0,0,0]
    return InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist)

def ForwardDynamics(thlist, thdotlist, tau, gvec, Ftip, Mlist, Glist, Slist):
    tau = np.array(tau)
    tau.shape = (np.size(tau),1)
    A = InertiaMatrix(thlist, Mlist, Glist, Slist)
    b = tau - CoriolisForces(thlist, thdotlist, Mlist, Glist, Slist) - GravityForces(thlist, gvec, Mlist, Glist, Slist) - EndEffectorForces(thlist, Ftip, Mlist, Glist, Slist)
    return np.linalg.solve(A,b)

def EulerStep(state, thdotdott, dt):
    state[0], state[1], thdotdott = np.array(state[0]), np.array(state[1]), np.array(thdotdott)
    state[0].shape, state[1].shape, thdotdott.shape = (np.size(state[0]), 1), (np.size(state[1]), 1), (np.size(thdotdott), 1)
    return state[0] + dt*state[1], state[1] + dt*thdotdott

def InverseDynamicsTrajectory(traj, Ftipset, gvec, Mlist, Glist, Slist):
    thlistset = traj[0]
    thdotlistset = traj[1]
    thdotdotlistset = traj[2]
    N = thlistset.shape[0]-1
    n = thlistset.shape[1]
    if not Ftipset:
        Ftip = [0,0,0,0,0,0]
        for i in range(N+1):
            Ftipset.append(Ftip)
    tauset = []
    for i in range(N+1):
        thlist = thlistset[i]
        thdotlist = thdotlistset[i]
        thdotdotlist = thdotdotlistset[i]
        Ftip = Ftipset[i]
        tau = InverseDynamics(thlist, thdotlist, thdotdotlist, gvec, Ftip, Mlist, Glist, Slist)
        tauset.append(tau)
    return np.reshape(tauset, (N+1,n))

def ForwardDynamicsTrajectory(initstate, tauset, dt, gvec, Ftipset, Mlist, Glist, Slist):
    initstate[0], initstate[1] = np.array(initstate[0]), np.array(initstate[1])
    initstate[0].shape, initstate[1].shape = (np.size(initstate[0]), 1), (np.size(initstate[1]), 1)
    thlistset = initstate[0].T
    thdotlistset = initstate[1].T
    N = tauset.shape[0]-1
    n = tauset.shape[1]    
    if not Ftipset:
        Ftip = [0,0,0,0,0,0]
        for i in range(N+1):
            Ftipset.append(Ftip)
    for i in range(1,N+1):
        thlist, thdotlist = thlistset[i-1], thdotlistset[i-1]
        tau = tauset[i-1].T
        Ftip = Ftipset[i-1]
        thdotdott = ForwardDynamics(thlist, thdotlist, tau, gvec, Ftip, Mlist, Glist, Slist)
        thlistdt, thdotlistdt = EulerStep([thlist, thdotlist], thdotdott, dt)
        thlistset, thdotlistset = np.vstack((thlistset, thlistdt.T)), np.vstack((thdotlistset, thdotlistdt.T))
    return [thlistset, thdotlistset]

def RobotModel(gvec, Ftip, Mlist, Glist, Slist):
    robot_model = []
    if not Ftip:
        Ftip = [0,0,0,0,0,0]
    robot_model.append(gvec)
    robot_model.append(Ftip)
    robot_model.append(Mlist)
    robot_model.append(Glist)
    robot_model.append(Slist)
    return robot_model

def FF_FB_Control(thlistd, thdotlistd, thdotdotlistd, thlist, thdotlist, Kp, Ki, Kd, robot_model):
    thlistd, thdotlistd, thdotdotlistd, thlist, thdotlist = np.array(thlistd), np.array(thdotlistd), np.array(thdotdotlistd), np.array(thlist), np.array(thdotlist)
    thlistd.shape, thdotlistd.shape, thdotdotlistd.shape, thlist.shape, thdotlist.shape = (np.size(thlistd),1), (np.size(thdotlistd),1), (np.size(thdotdotlistd),1), (np.size(thlist),1), (np.size(thdotlist),1)
    e = thlistd - thlist
    edot = thdotlistd - thdotlist
    thdotdotlist = thdotdotlistd + matmult(Kp,e) + matmult(Kd,edot) + Ki
    return InverseDynamics(thlist, thdotlist, thdotdotlist, robot_model[0], robot_model[1], robot_model[2], robot_model[3], robot_model[4])

def SimulateControl(ac_robot_model, es_robot_model, Kp, Ki, Kd, initstate, trajd, dt, plot=False):
    N = robotraj[0].shape[0]
    thlist2 = initstate[0]
    thlist2 = np.array(thlist2)
    thlist2.shape = (np.size(thlist2),1)
    n = thlist2.shape[0]
    thdotlist2 = initstate[1]
    eint=0
    Ki0 = Ki
    thlistset=[]
    thlistset.append(thlist2)
    comtauset=[]
    for i in range(N):        
        thlistd = trajd[0][i,:]
        thdotlistd = trajd[1][i,:]
        thdotdotlistd = trajd[2][i,:]
        thlist = np.copy(thlist2)        
        thdotlist = thdotlist2
        thlistd = np.array(thlistd)
        thlist.shape, thlistd.shape = (np.size(thlist),1), (np.size(thlistd), 1)
        e = thlistd-thlist
        eint += e*dt
        Ki = matmult(Ki0, eint)         
        comtau = FF_FB_Control(thlistd, thdotlistd, thdotdotlistd, thlist, thdotlist, Kp, Ki, Kd, es_robot_model)
        comtauset.append(comtau)         
        thdotdotlist = ForwardDynamics(thlist, thdotlist, comtau, ac_robot_model[0], ac_robot_model[1], ac_robot_model[2], ac_robot_model[3], ac_robot_model[4])
        thlist2, thdotlist2 = EulerStep([thlist, thdotlist], thdotdotlist, dt)
        thlistset.append(thlist2)  
    del thlistset[-1]
    thlistset = np.reshape(thlistset, (N,n))
    comtauset = np.reshape(comtauset, (N,n))
    if plot:
        x = np.linspace(0,int(N*dt),N)
        plt.figure()
        plt.plot(x, thlistset[:,0], label='theta - joint 1')
        plt.plot(x, thlistset[:,1], label='theta - joint 2')
        plt.plot(x, thlistset[:,2], label='theta - joint 3')
        plt.plot(x, thlistset[:,3], label='theta - joint 4')
        plt.plot(x, thlistset[:,4], label='theta - joint 5')
        plt.plot(x, thlistset[:,5], label='theta - joint 6')
        plt.plot(x, trajd[0][:,0], label='desired theta')
        plt.xlabel('time (s)')
        plt.ylabel('joint angles')
        plt.title('Plot of actual vs desired joint angles as a function of time')
        plt.legend(loc='upper left')
        plt.show()    
    return comtauset, thlistset