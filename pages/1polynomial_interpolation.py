import streamlit as st
import numpy as np
import pandas as pd
import re
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from PIL import Image
st.sidebar.markdown("# Polynomial Interpolation")

tab1, tab2, tab3,tab4=st.tabs(["Overview", "Monomial interpolation", "Lagrange","Newton"])

with tab1:
    st.markdown('''### Interpolation Problem Statement''')
    st.markdown('''
    > Given a set of data points($x_i,y_i$), we want to find a n-th degree polynomial $P_n(x)=c_0+c_1x+...+c_nx^n$ that goes through our data points. 
    
    ### Applications
    * **Data fitting** (discrete): given data($x_i,y_i$),find a reasonable function v(x) that fits the data.
    * **Approximate functions** (continuous): for a complicated function f(x), find a simpler function v(x) that approximates f. By replacing f(x) with v(x),  the value of derivative at some point and definite integral of a function can be computed easily. 
    
    ### Different polynomial interpolation
    1. **Monomial interpolation**: $P_n(x)=c_0+c_1x+...+c_nx^n$
    1. **Lagrange Interpolation**: $P_n(x)=y_0L_0(x)+...+y_nL_n(x)$
    1. **Newton's**: $P_n(x)=c_0+c_1(x-x_0)+c_2(x-x_0)(x-x_1)+...+c_n(x-x_0)(x-x_1)...(x-x_{n-1})$''')

with tab2:
    st.markdown('''
        For any n + 1 points, $(x_0,y_0),(x_1,y_1)...(x_n,y_n)$ with distinct $x_i$, there exists a unique polynomial $P_n$ of degree n such that $P_n(x_i)=y_i$ for i=0,1...n.
        
        In other words, there is a solution for the following:''')
    st.latex(r'''\begin{pmatrix}
1 & x_0 & \cdots & {x_0}^n \\
1 & x_1 & \cdots & {x_1}^n \\
\vdots  & \vdots  & \ddots & \vdots  \\
1 & x_n & \cdots & {x_n}^n
\end{pmatrix}
\begin{pmatrix}
c_0 \\
c_1 \\
\vdots \\
c_n
\end{pmatrix}=\begin{pmatrix}
y_0 \\
y_1 \\
\vdots \\
y_n
\end{pmatrix}''')
    st.markdown('''The determinant of the coefficient matrix is known as the **Vandermonde determinant**. Since the Vandermonde determinant is nonzero, the system of equations has a unique solution.''')
    with st.expander("A simple example for monomial interpolation"):
        st.markdown('''
        For any n + 1 points, $(x_0,y_0),(x_1,y_1)...(x_n,y_n)$ with distinct $x_i$, there exists a unique polynomial $P_n$ of degree n such that $P_n(x_i)=y_i$ for i=0,1...n.
        
        Let us illustrate how to solve the problem using a simple example: find a quadratic polynomial $P_2(x)=c_0+c_1x+c_2x^2$ that passes through (3,32),(5,74), and (7,132).
        
        We can come up with the following system of linear equations:
        * $c_0+3c_1+9c_2=32$
        * $c_0+5c_1+25c_2=74$
        * $c_0+7c_1+49c_2=132$
        ''')
        st.latex(r'''\begin{pmatrix}
            1 & 3 & 9 \\
            1 & 5 & 25 \\
            1 & 7 & 49
            \end{pmatrix}  \begin{pmatrix}
            c_0 \\
            c_1 \\
            c_2
            \end{pmatrix} = \begin{pmatrix}
            32 \\
            74 \\
            132
            \end{pmatrix}''')
        code='''import numpy as np
A= np.array([[1, 3,9], [1, 5,25],[1,7,49]])
b=np.array([32,74,132])
x = np.linalg.solve(A, b)
print(x)
        '''
        st.code(code,language='Python')
        st.markdown("Solving the linear system of equations will yield $c_0=-1,c_1=5,c_2=2.$")
    
    with st.expander("Pros and Cons of Monomial Interpolation"):
        st.markdown(r'''
        To obtain the interpolating polynomial $P_n(x)$, we need to form the Vandermonde matrix and solve the linear system.
        * Advantage: simplicity
        * Disadvantage: Vandermonde matrix is often ill-conditioned. Using this approach takes $\frac{2}{3}n^3$.''')
    
    with st.expander("I/O"):
        x_values=st.text_input("Enter your x values(separated by comma)",'3,5,7')
        y_values=st.text_input("Enter your y values(separated by comma)",'32,74,132')
        x=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', x_values)))
        y=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', y_values)))
        if len(x)==len(y) and len(x)!=0:
            co1,co2,co3,co4=st.columns((1,1,2,2))
            with co1:
                st.write("x values:",x)
            with co2:
                st.write("y values:",y)
            with co3:
                st.write("Vandermonde matrix")
                A=np.ones(len(x))
                for i in range(len(x)-1):
                    temp=np.array([j**(i+1) for j in x])
                    A=np.vstack((A,temp))
                A=A.T
                st.write(A)
            with co4:
                c_values=np.linalg.solve(A,y)
                Pn=Polynomial(c_values)
                xrange=np.arange(min(x)-1,max(x)+1,0.5)
                fig,ax=plt.subplots()
                ax.plot(xrange,Pn(xrange),'b',np.array(x),np.array(y),'ro')
                st.pyplot(fig)
            st.write(r'''Solving for $P_n(x)=c_0+c_1x+...+c_nx^n$, we have''',c_values)
        elif len(x)==0 or len(y)==0:
            st.error("input is missing")
        else:
            st.error("len(x) and len(y) not equal")
    
with tab3:
    st.markdown('''We can also use Lagrange polynomials for interpolation:''')
    st.latex(r'''L_j(x)=\frac{(x-x_0)...(x-x_{j-1})(x-x_{j+1})...(x-x_n)}{(x_j-x_0)...(x_j-x_{j-1})(x_j-x_{j+1})...(x_j-x_n)}''')
    st.markdown('''The formula is for **Lagrange multiplier**. Applying the formula, we can identify basic building blocks for Lagrange polynomials and come up with our interpolant $P(x)=\sum_{i=0}^{n}y_iL_i(x)$.''')
    
    with st.expander("A simple example"):
        st.markdown('''Solving $P_2(x)=c_0+c_1x+c_2x^2$ for P(3)=32, P(5)=74, and P(7)=132.''')
        st.markdown(r'''$L_0(x)=\frac{(x-5)(x-7)}{(3-5)(3-7)}$ with $L_0(5)=L_0(7)=0$ and $L_0(3)=1$''')
        st.markdown(r'''$L_1(x)=\frac{(x-3)(x-7)}{(5-3)(5-7)}$ with $L_1(3)=L_1(7)=0$ and $L_1(5)=1$''')
        st.markdown(r'''$L_2(x)=\frac{(x-3)(x-5)}{(7-3)(7-5)}$ with $L_2(3)=L_2(5)=0$ and $L_2(7)=1$''')
        st.latex(r'''P(x)=32L_0(x)+74L_1(x)+132L_2(x)''')
        st.latex(r'''=32\frac{(x-5)(x-7)}{8}+74\frac{(x-3)(x-7)}{-4}+132\frac{(x-3)(x-5)}{8}
        =2x^2+5x-1''')
        cd='''from scipy.interpolate import lagrange
f = lagrange(x, y)'''
        st.code(cd,language="python")
    with st.expander("I/O"):
        cl1,cl2=st.columns(2)
        with cl1:
            x_vals=st.text_input("Enter your x values(separated by comma)",key=5)
            y_vals=st.text_input("Enter your y values(separated by comma)",key=6)
            x=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', x_vals)))
            y=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', y_vals)))
            number = st.number_input('Enter a number for x',step=0.01)
            st.write('The current x is ', number)
        with cl2:
            if len(x)==len(y) and len(x)!=0:
                fig,ax=plt.subplots()
                f=lagrange(x, y)
                xran=np.arange(min(min(x),number)-1,max(max(x),number)+1,0.5)
                ax.plot(xran,f(xran),'b',x,y,'ro')
                st.pyplot(fig)
                st.write("when x=",number,",y=",f(number))
            elif len(y)==0 or len(x)==0:
                st.error("missing input")
            else:
                st.error("len(x) and len(y) not equal")



with tab4:
    st.markdown('''Newton's form of the interpolating polynomial is especially elegant, allowing
the polynomial to evolve, increasing in degree as it satisfies an increasing number
of the desired criteria.''')
    st.markdown(r'''The n-th degree polynomial P(x) with $P(x_i)=f(x_i)$ is written in the Newton's form as:
    $P(x)=A_0+\sum_{k=1}^{n}A_k(x-x_0)(x-x_1)...(x-x_{k-1})$''')
    st.markdown('''The special feature of the Newton’s polynomial is that the coefficients $A_i$ can be determined using a very simple mathematical procedure.''')
    st.latex(r'''A_0=f(x_0); A_1=\frac{f(x_1)-f(x_0)}{x_1-x_0}; A_2=\frac{\frac{f(x_2)-f(x_1)}{x_2-x_1}-\frac{f(x_1)-f(x_0)}{x_1-x_0}}{x_2-x_0}''')
    img=Image.open('pages/divided_diff.png')
    st.image(img,caption="Divided Difference")
    st.markdown(r'''Given a sequence of distinct points $x_i$ and function values $f(x_i)$, the kth divided difference of f at $x_j$, denoted by $f[x_j,x_{j+1},...,x_{j+k}]$, can be determined by:''')
    st.latex(r'''f[x_j,x_{j+1},...,x_{j+k}]=\frac{f[x_{j+1},x_{j+2},...,x_{j+k}]-f[x_j,x_{j+1},...,x_{j+k-1}]}{x_{j+k}-x_j}''')
    
    with st.expander("A simple example"):
        img1=Image.open('pages/divided_diff_ex.png')
        st.image(img1,caption="Applying Divided Difference")
        cds='''def _poly_newton_coefficient(x, y):    
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a

def newton_polynomial(x_data, y_data, x):
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p
    return p'''
        st.code(cds,language="python")
    with st.expander("I/O"):
        cl3,cl4=st.columns(2)
        with cl3:
            x_vals=st.text_input("Enter your x values(separated by comma)",key=3)
            y_vals=st.text_input("Enter your y values(separated by comma)",key=4)
            x=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', x_vals)))
            y=list(map(int,re.findall(r'-?\d*\.{0,1}\d+', y_vals)))
            num = st.number_input('Enter a number for x',step=0.02)
            st.write('The current x is ', num)
        def _poly_newton_coefficient(x, y):    
            m = len(x)
            x = np.copy(x)
            a = np.copy(y)
            for k in range(1, m):
                a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
            return a

        def newton_polynomial(x_data, y_data, x):
            a = _poly_newton_coefficient(x_data, y_data)
            n = len(x_data) - 1  # Degree of polynomial
            p = a[n]
            for k in range(1, n + 1):
                p = a[n - k] + (x - x_data[n - k])*p
            return p
        with cl4:
            if len(x)==len(y) and len(x)!=0:

                fig,ax=plt.subplots()
                
                xran=np.arange(min(min(x),num)-1,max(max(x),num)+1,0.5)
                ax.plot(xran,newton_polynomial(x,y,xran),'b',x,y,'ro')
                st.pyplot(fig)
                st.write("when x=",num,",y=",newton_polynomial(x,y,num))
            elif len(y)==0 or len(x)==0:
                st.error("please enter some input")
            else:
                st.error("len(x) and len(y) not equal")