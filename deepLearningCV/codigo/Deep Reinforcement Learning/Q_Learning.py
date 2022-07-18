#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:41:55 2022

@author: franciscorealescastro
"""
import numpy as np
import random

maze=np.array([[0,0,1],[-100,-100,0],[0,0,100]])


m,n=maze.shape

actions_space_size=4

Q_table=np.zeros((m,n,actions_space_size))

state=np.array([0,0])
acciones_sting=np.array(["izquierda","derecha","arriba","abajo"])

def step(a,s):
    done=False
    if a==0:
        s-=np.array([0,1])
    if a==1:
        s+=np.array([0,1])
    if a==2:
        s-=np.array([1,0])
    if a==3:
        s+=np.array([1,0])
    
    if s[0]<0:
        s[0]=0
    if s[1]<0:
        s[1]=0
    if s[0]==m:
        s[0]-=1
    if s[1]==n:
        s[1]-=1
    
    r=maze[s[0],s[1]]
    if r==100 or r==-100:
        done=True
    return s,r,done

steps=1000
eps=1
eps_dec=0.99
alpha=0.9
gamma=0.9
random.seed(0)
for _ in range(0,steps):
    if eps>random.random():
        accion=random.randint(0,3)
        
        state_t,reward,done=step(accion,state.copy())
        Q_table[state[0],state[1],accion] \
        = (1-alpha)*Q_table[state[0],state[1],accion]+alpha*(reward + gamma*np.max(Q_table[state_t[0],state_t[1],:]))
    else:
        accion=np.argmax(Q_table[state[0],state[1],:])
        accion=np.int8(accion)
        state_t,reward,done=step(accion,state)
        Q_table[state[0],state[1],accion] \
        = (1-alpha)*Q_table[state[0],state[1],accion]+alpha*(reward + gamma*np.max(Q_table[state_t[0],state_t[1],:]))
    if not(done):
        state=state_t.copy()
    else:
        state[0],state[1]=0,0
        reward=maze[0,0]
        done=False
        eps*=eps_dec

    
value=np.max(Q_table,axis=2)  

policy=np.argmax(Q_table,axis=2)
policy[maze==100]=-1
policy[maze==-100]=-1
policy_string=acciones_sting[policy]
policy_string[maze==100]='Win'
policy_string[maze==-100]='x'
        