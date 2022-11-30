import streamlit as st
import numpy as np
import pandas as pd

tab1,tab2,tab3=st.tabs(['Overview','Proofs','Other Polynomials'])

with tab1:

    st.markdown(r'''
    Let $P_n(x)=c_0\phi_0+c_1\phi_1+...+c_n\phi_n$ be a polynomial that converges uniformly to a given function f(x) on [a,b].
    The **Weierstrass Approximation Theorem** states that $||f(x)-P_n(x)||<\epsilon$ for any $\epsilon>0$. 
    
    We will provide several different proofs to Weierstrass Approximation, including
    1. Fourier series
    2. Landau kernel
    3. Lebesgue
    4. Bernstein polynomials''')
