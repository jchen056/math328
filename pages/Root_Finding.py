import streamlit as st
import numpy as np
import pandas as pd

st.sidebar.markdown("### Root Finding")

st.markdown(r'''We will design algorithms to approximate solutions to nonlinear scalar equations f(x)=0 where the variable x runs in an interval [a,b].''')

tab1,tab2,tab3=st.tabs(['Bisection method','Fixed Point Iteration',"Newton's method"])
with tab1:
    st.markdown('''Let f be continuous on [a,b], satisfying f(a)f(b)<0. Then f has a root between a and b. In other words, there exists a number $x^*$ in [a,b] so that $f(x^*)=0$ by **intermediate value theorem**.''')
    st.markdown(r'''
    Bisection method has a solution error of $|x^*-x_n|< \frac{b-a}{2^{n+1}}$ where n is the number of iterations.
    * A solution is correct within p decimal places if the error is less than $0.5*10^{-p}$
    * Generally speaking, every 4 iterations stabilize a digit for Bisection method.''')
    

    codes_bis='''def bisection(a,b,atol):
    f=lambda x: x**3-30*x*x+2552#function which you want to find roots
    if f(a)*f(b)>0:
        print("there is probably no solution")
        return
    
    n=math.ceil(math.log2(b-a)-math.log2(2*atol))
    print("number of iterations:",n)
    for i in range(n):
        c=(a+b)/2
        if f(a)*f(c)<0:
            b=c
        else:
            a=c
    return c'''
    st.code(codes_bis,language='python')
with tab2:
    st.markdown(r'''
    The real number $x^*$ is a fixed point of g if $x^*=g(x^*)$. 
    
    For FPI, we have our initial guess $x_0$ and $x_{k+1}=g(x_k)$. 
    
    Let g(x) be continuously differentiable and $x^*=g(x^*)$ is a fixed point. If $|g'(x)| \le \rho <1$, then FPI converges linearly to $x^*$ with the rate $\log_{10}\rho$. On the other hand, $x^*$ is an unstable fixed point if $|g'(x^*)|>1$.
   
    $e_i=|x_i-x^*|=|g(x_{i-1})-g(x^*)|=|g'(x_i-1)(x_{i-1}-x^*)|=\rho e_{i-1}$
    
    Iterating, we have $e_i= {\rho}^i e_0.$
    ''')
    codes_fpi='''def FPI(x0,n):#x0 is your initial guess; n is the number of iterations
    g=lambda x:math.log(7-x)#enter your x=g(x) here
    x=g(x0)#FPI for the first time
    for i in range(1,n):
        x=g(x)
    return x'''
    st.code(codes_fpi,language='python')
with tab3:
    st.markdown(r'''
    Suppose that f is twice differentiable and f(r)=0. If $f'(r)\ne 0$, then Newton's method $x_{n+1}=x_k-\frac{x_k}{f'(x_k)}$ is locally quadratically convergent to $x^*$.''')
    codes_newton='''def newton_method(x0,n):
    f=lambda x: x**3-5
    fp=lambda x: 3*x*x
    
    x=x0-f(x0)/fp(x0)
    for i in range(1,n):
        x=x-f(x)/fp(x)
        print(i+1,x)
    return x'''
    st.code(codes_newton,language='python')

    st.markdown(r'''
    **Secant method:**
    
    Suppose $x_0,x_1$ are our initial guesses. We have $x_{k+1}=x_k-\frac{f(x_k)(x_k-x_{k-1})}{f(x_k)-f(x_{k-1})}$.''')
    
    codes_sec='''def secant_method(x0,x1,n):#secant method require two initial condition x0, x1; n is the number of iterations
    
    f=lambda x: x**3-3*x+2#the function
    
    for i in range(n):
        fx1=f(x1)
        fx0=f(x0)
        xtemp=x1
        x1=x1-fx1*(x1-x0)/(fx1-fx0)
        x0=xtemp
    return x1'''
    st.code(codes_sec,language='python')