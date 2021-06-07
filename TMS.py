# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 20:16:29 2021

@author: viraj
"""

import numpy as np
from mayavi.mlab import*
import sympy as sym
import matplotlib.pyplot as plt

Radius = 45e-3
Turns = 10
Pitch = 10e-3
wd = 1e-3
dR = wd + 0.2e-3
points = np.linspace(0,Turns,360*Turns)
s = np.array([0,-10e-3,0])
time = np.linspace(0,1e-3,1001)
L = 24.5e-6
C = 10e-6
r_l = 25.8e-3 # coil resistance
r_c = 300e-3 # circuit resistance
R = r_l + r_c
V0 = 5e3

def generate_coil(Radius,Turns,Pitch,dR,points,wd):
    t = sym.symbols('t')
    C_x = (Radius + dR*t)*sym.cos(2*sym.pi*t)
    C_y = Pitch*t
    C_z = (Radius + dR*t)*sym.sin(2*sym.pi*t)
    Cx = sym.lambdify(t,C_x)
    Cy = sym.lambdify(t,C_y)
    Cz = sym.lambdify(t,C_z)
    Coil = np.array([Cx(points),Cy(points),Cz(points)])            
    l = plot3d(Coil[0],Coil[1],Coil[2],tube_radius = wd/2)
    return [l,Coil]

def generate_current(L,C,R,V0,time):
    t = time
    a = R/(2*L) # damping coefficient
    w_o = 1/np.sqrt(L*C)
    I = np.zeros(len(t))
    dI_dt = np.zeros(len(t))
    dc = a/w_o

    if dc > 1:
        A1 = V0*(2*np.sqrt(a**2 - w_o**2))**-1
        A2 = -A1
        for i in range(len(t)):
            I[i] = A1*np.exp((-a + np.sqrt(a**2 - w_o**2))*t[i]) + A2*np.exp((-a - np.sqrt(a**2 - w_o**2))*t[i])
            dI_dt[i] = A1*(-a + np.sqrt(a**2 - w_o**2)*np.exp(t[i]*(-a + np.sqrt(a**2-w_o**2)))) + A2*(-a - np.sqrt(a**2 - w_o**2)*np.exp(t[i]*(-a - np.sqrt(a**2-w_o**2))))
    elif dc == 1:
        D1 = V0/L
        D2 = 0
        for i in range(len(t)):
            I[i] = D1*t[i]*np.exp(-a*t[i])
            dI_dt[i] = -D1*a*t[i]*np.exp(-a*t[i]) + D1*np.exp(-a*t[i]) - D2*a*np.exp(-a*t[i])
    else:
        w_d = np.sqrt(w_o**2 - a**2)
        B1 = 0
        B2 = V0/(L*w_d)
        for i in range(len(t)):
            I[i] = B2*np.exp(-a*t[i])*np.sin(w_d*t[i])
            dI_dt[i] = -B1*a*np.exp(-a*t[i])*np.cos(t[i]*w_d) - B2*a*np.exp(-a*t[i])*np.sin(t[i]*w_d) -B1*w_d*np.exp(-a*t[i])*np.sin(t[i]*w_d) + B2*w_d*np.exp(-a*t[i])*np.cos(t[i]*w_d)
    Imax, tmax = (np.max(I),np.argmax(I))
    plt.plot(t, I)
    plt.figure()
    plt.plot(t,dI_dt)
    return [I,dI_dt,Imax,tmax]

def generate_fields(Radius,Pitch,Turns,s,points,Imax,dI_dt):
    mu = 1.25e-6
    K = Imax*mu/(4*np.pi)
    h = float(points[1]-points[0])
    t = sym.symbols('t')
    C_x = (Radius + dR*t)*sym.cos(2*sym.pi*t)
    C_y = Pitch*t
    C_z = (Radius + dR*t)*sym.sin(2*sym.pi*t)
    dC_x = C_x.diff(t)
    dC_y = C_y.diff(t)
    dC_z = C_z.diff(t)
    r = sym.sqrt((s[0] - C_x)**2 + (s[1]-C_y)**2 + (s[2]- C_z)**2)
    dA_x = dC_x/r
    dA_y = dC_y/r
    dA_z = dC_z/r
    dAx = sym.lambdify(t, dA_x)
    dAy = sym.lambdify(t, dA_y)
    dAz = sym.lambdify(t, dA_z)
    A = K*h*np.array([np.sum(dAx(points)),np.sum(dAy(points)),np.sum(dAz(points))])
    E = np.empty([len(dI_dt),len(A)])
    for i in range(len(dI_dt)):
        E[i] = -dI_dt[i]*A/Imax
    plt.figure()
    plt.plot(time,E[:,1])    
    return [A,E]   

Coil = generate_coil(Radius,Turns,Pitch,dR,points,wd)[1]
I,dI_dt,Imax,tmax = generate_current(L,C,R,V0,time)

E = generate_fields(Radius,Pitch,Turns,s,points,Imax,dI_dt)[1]


