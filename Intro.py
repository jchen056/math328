import streamlit as st

st.markdown("# All about the Approximation Theory")

st.markdown('''The report showcases what I have learned in the independent study about the **Approximation Theory** with Professor Bak.

We will begin with **polynomial interpolation**. Given n+1 data points, we want to find an interpolating polynomial of degree n to approximate the function f(x). As building blocks, polynomial interpolants can be used for **differentiation**, **integration**, and solutions of **differential equations**. We will demonstrate the application of polynomial interpolation in integration.

Moving on, we will review some linear algebra concepts because linear algebra helps unify much of numerical analysis. The unique aspects of approximation in **inner product spaces** will lead us to the topic of **Fourier series** and ultimately a proof to **Weierstrass' Theorem**.

Furthermore, we will talk about other possible proofs to Weierstrass' theorem, using Chebyshev polynomials and Bernstein polynomials.''')
