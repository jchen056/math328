import streamlit as st
import numpy as np
import pandas as pd

tab1,tab2,tab3=st.tabs(['Overview','Other Polynomials','Proofs'])

with tab1:

    st.markdown(r'''
    Let $P_n(x)=c_0\phi_0+c_1\phi_1+...+c_n\phi_n$ be a polynomial that converges uniformly to a given function f(x) on [a,b].
    The **Weierstrass Approximation Theorem** states that $||f(x)-P_n(x)||<\epsilon$ for any $\epsilon>0$. 
    
    We will provide several different proofs to Weierstrass Approximation, including
    1. Fourier series
    2. Landau kernel
    3. Lebesgue
    4. Bernstein polynomials''')

with tab2:
    with st.expander("Osculating Polynomials and Hermite Polynomials"):
        st.markdown("### Osculating Polynomials")
        st.markdown('''
        **Osculating polynomials** generalize both Taylor polynomials and the Lagrange polynomials. Assume we are given n+1 distinct numbers, $x_0,x_1,...,x_n$, such that $x_i \in[a,b]$, and integers $m_0,m_1,...,m_n$, such that $m_i\ge0$ and $m=\max(m_0,m_1,...,m_n)$.
        
        The osculating polynomial approximating a function $f(x)\in C^m[a,b]$ at $x_i$ for each $i\in\{0,1,...,n\}$ is the polynomial of the least degree that has the same values as the function f(x) and all its derivatives of order less than or equal to $m_i$ at each $x_i$. The degree of this osculating polynomial is at most $M=\sum_{i=0}^{n}m_i+n$.''')
        
        st.markdown(r'''Let $x_0,x_1,...,x_n$ be n+1 distinct numbers in $[a,b]$ ($i\in \{0,1,...,n\}$) and 
        let $m_i \ge 0$ and $m_i \in Z$. Suppose that $f(x) \in C^m [a,b]$. The osculating polynomial approximating f(x) is the polynomial P(x)
of the least degree such that $\frac{d^kP(x_i)}{dx^k}=\frac{d^kf(x_i)}{dx^k}$ for each $i \in \{0,1,...,n\}$ and $k\in \{0,1,...,m\}$.
        ''')
        st.markdown("### Hermite Polynomials")
        st.markdown(r'''
        **Hermite Polynomials** are those osculating polynomials for which $m_i=1$ for all i.  
        
        For a given function f(x), the polynomials agree with f(x) at $x_0,x_1,...,x_n$. In addition, since their first derivatives agree with those of f(x), 
        we have the same shape as the functions at $(x_i,f(x_i))$. In other words, *we fit the points and the slopes and no higher derivatives*.''')
