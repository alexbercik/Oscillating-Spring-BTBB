#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:18:39 2021

@author: bercik
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# Define cnstants
g=9.81
k=1
m=1
lam = np.sqrt(k/m)

# Define initial conditions, time step, and final solution time
y0 = 0
v0 = 0
h = 0.1
tfinal = 10

# Analytical (exact) solution
c1 = v0
c2 = y0+g*m/k
t_ex = np.arange(0,tfinal,0.01)
y_ex = c1*np.sin(lam*t_ex) + c2*np.cos(lam*t_ex) - g*m/k

# Numerical solution
t = np.arange(0,tfinal,h)

def a(y):
    ''' acceleration: defines the ODE '''
    return -(k/m)*y -g 

def exp_euler(y,v):
    ''' UNSTABLE explicit euler method '''
    v_new = v + h * a(y)
    y_new = y + h * v
    return y_new, v_new

def mod_euler(y,v):
    ''' STABLE explicit euler method '''
    v_new = v + h * a(y)
    y_new = y + h * v_new
    return y_new, v_new

# Initialize numerical solutions with initial conditions
y_exp, v_exp, y_mod, v_mod = np.zeros(t.shape), np.zeros(t.shape), np.zeros(t.shape), np.zeros(t.shape)
y_exp[0], y_mod[0] = y0, y0
v_exp[0], v_mod[0] = v0, v0

# Match through time and calculate the next time step
for i in range(len(t)-1):  
     y_exp[i+1], v_exp[i+1] = exp_euler(y_exp[i], v_exp[i])
     y_mod[i+1], v_mod[i+1] = mod_euler(y_mod[i], v_mod[i])

# Plot the final solutions
plt.figure(figsize=(6,4))
plt.plot(t_ex, y_ex, label='Exact')
plt.plot(t, y_exp, label='Euler')
plt.plot(t, y_mod, label='Mod Euler')
plt.legend(loc='lower left',fontsize=13)
plt.xlabel(r'$t$',fontsize=14)
plt.ylabel(r'$y$',rotation=0,fontsize=13)
plt.title(r'Oscillating Spring',fontsize=16)
plt.tight_layout()
plt.show()