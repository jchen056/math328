import streamlit as st
import numpy as np
import pandas as pd
import re
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from PIL import Image
st.sidebar.markdown("# Polynomial Interpolation")

tab1, tab2, tab3,tab4,tab5=st.tabs(["Overview", "Monomial interpolation", "Lagrange","Newton","Applications"])

with tab1:
    st.markdown('''### Interpolation Problem Statement''')
    st.markdown('''
    > Given a set of data points($x_i,y_i$), we want to find a n-th degree polynomial $P_n(x)=c_0\phi_0+c_1\phi_1+...+c_n\phi_n$ that goes through our data points. 
    
    ### Applications
    * **Data fitting** (discrete): given data($x_i,y_i$),find a reasonable function v(x) that fits the data.
    * **Approximate functions** (continuous): for a complicated function f(x), find a simpler function v(x) that approximates f. By replacing f(x) with v(x),  the value of derivative at some point and definite integral of a function can be computed easily. 
    
    ### Different polynomial interpolation
    1. **Monomial interpolation**: $P_n(x)=c_0+c_1x+...+c_nx^n$
    1. **Lagrange Interpolation**: $P_n(x)=y_0L_0(x)+...+y_nL_n(x)$
    1. **Newton's**: $P_n(x)=c_0+c_1(x-x_0)+c_2(x-x_0)(x-x_1)+...+c_n(x-x_0)(x-x_1)...(x-x_{n-1})$
    
    Monomial is simple but suffers when the Vandermonde matrix is ill-conditioned. Lagrange is the most stable but it is not adaptive; when a new data point is added, you have to start over your computation. Newton's is adaptive. ''')

    st.markdown(r'''
    ### Polynomial Interpolation Error
    
    The error in the **Taylor polynomial** (only works for intepolation, not for extrapolation): 
    * $P(x)=\sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$
    * $f(x)-P_n(x)=\frac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)(x-x_1)...(x-x_n)$
    * We have error bound $\max_{a \le x \le b}|f(x)-P_n(x)|\le \frac{1}{(n+1)!}\max_{a\le t \le b}|f^{n+1}(t)|\max_{a\le s\le b}|s-x_i|$.
    
    In reality, $f^{(n+1)}(\xi)$ is hard to compute given the data points. Even if the function f(x) is given, finding its derivative can be a headache.

    It is not true that higher degree will lead to higher accuracy. Low order approximations are often reasonable. High-degree interpolants, on the other hand, are expensive to compute and can be poorly behaved.''')


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
        st.code(code,language='python')
        st.markdown("Solving the linear system of equations will yield $c_0=-1,c_1=5,c_2=2.$")

        st.markdown('''**Food for Thought**: Can you always find a unique solution to the system of equations?''')
        st.markdown('''Trying to find the nullspace for the Vandermonde matrix in our simple example, we have''')
        st.latex(r'''\begin{pmatrix}
            1 & 3 & 9 \\
            1 & 5 & 25 \\
            1 & 7 & 49
            \end{pmatrix}  \begin{pmatrix}
            c_0 \\
            c_1 \\
            c_2
            \end{pmatrix} = \begin{pmatrix}
            0 \\
            0 \\
            0
            \end{pmatrix}''')
        st.markdown(r'''We will get $c_0=0, c_1=0,c_2=0$ because a quadratic function has either no roots or two roots.''')
        st.markdown('''Therefore, **Nullity** of our Vandermonde matrix is 0 and **rank** of our our Vandermonde matrix will be the number of independent columns. Thus, our Vandermonde matrix is **nonsingular** and we have a unique solution.''')
    
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

with tab5:
    st.markdown('''As mentioned previously, polynomial interpolation are the building blocks for more complex algorithms in differentiation, integration, and solutions of differential equations.''')
    with st.expander("Numerical Differentiation"):
        st.markdown('''* Two-point formula:''')
        st.latex(r'''f'(x_0)=\frac{f(x_0+h)-f(x_0)}{h}+\frac{h}{2}f''(\xi)''')
        st.markdown('''* Three-point midpoint formula:''')
        st.latex(r'''f'(x_0)=\frac{f(x_0+h)-f(x_0-h)}{2h}+\frac{h^2}{6}f^{(3)}(\xi)''')
        st.markdown('''* Five-point midpoint formula''')
        st.latex(r'''f'(x_0)=\frac{1}{12h}*[f(x_0-2h)-8f(x_0-h)+8(x_0+h)-f(x_0+2h)]+\frac{h^4}{30}f^{(4)}(\xi)''')

    with st.expander('''Numerical Integration'''):
        int1,int2=st.columns(2)
        # with int1:
        #     img_r=Image.open('Riemann.jpg')
        #     st.image(img_r,caption="Riemann Sum")
        # with int2:
        #     img_s=Image.open('Simpson.png')
        #     st.image(img_s,caption="Simpson's Rule")
        st.markdown('''In practice, finding an exact solution for the integral of a function is difficult or impossible. Instead, we seek approximations of the definite integral for a given finite interval [a,b] and the function f.''')
        st.latex(r''' \int_{x=a}^{x=b} f(x)dx\approx \sum_{j=1}^{n}a_jf(x_j)''')
        st.markdown('''
        The numerical integration formula, often referred to as a **quadrature rule**, has abscissae $x_j$ and weights $a_j$.
        
        Using polynomial intepolation, we have:''')
        st.latex(r''' \int_{x=a}^{x=b} f(x)dx\approx \int_{x=a}^{x=b} P_n(x)dx''')
        st.markdown(r'''
        Using the Lagrange interpolating polynomial, we have 
        
        $\int_{x=a}^{x=b} f(x)dx\approx \int_{x=a}^{x=b} \sum_{a}^{b} f(x_i)L_i(x)dx+\int_{x=a}^{x=b}\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^{n}(x-x_i)dx$
        
        $=\int_{a}^{b} f(x)dx\approx \int_{a}^{b} a_if(x_i)+\int_{x=a}^{x=b}\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^{n}(x-x_i)dx$''')
        st.markdown(r'''
        Suppose we have n strips and n+1 points. $h=x_{i+1}-x_i=\frac{b-a}{n}$
        1. **Riemann Integral**
        $\int_{x=a}^{x=b} f(x)dx\approx \sum_{i=0}^{n-1}hf(x_i)$ (left-endpoint)

        $\int_{x=a}^{x=b} f(x)dx\approx \sum_{i=1}^{n}hf(x_i)$ (right-endpoint)

        $\int_{x=a}^{x=b} f(x)dx\approx \sum_{i=0}^{n-1}hf(y_i)$ where $y_i=\frac{x_i+x_{i+1}}{2}$ (midpoint)''')
        codes_Riemann='''#compute the area under the curve f(x) on [a,b] using numerical integration
def f_circle(x):
  return np.sin(x)

#[a,b] is your interval; n is the number of strips

def Riemann_sum(f_circle,a,b,n):
  h=(b-a)/n #stepsize
  x_values=np.arange(start=a,stop=b+h,step=h)
  y_values=[f_circle(i) for i in x_values]
  left=h*np.sum(y_values[0:-1])
  right=h*np.sum(y_values[1:])

  x_midpoints=(x_values[0:-1]+x_values[1:])/2
  midpoint=h*(np.sum(f_circle(i) for i in x_midpoints))
                     
  return left, right, midpoint
Riemann_sum(f_circle,0,np.pi,10)'''
        st.code(codes_Riemann,language='python')

        st.markdown(r'''
        2. **Trapezoid Rule**:
        **Trapezoid Rule**: $\int_{x=a}^{x=b} f(x)dx\approx \frac{h}{2}(f(x_0)+2f(x_1)-\frac{h^3}{12}f''(\xi)$ where $h=x_1-x_0$

        **Composite Trapezoid Rule**: $\int_{x=a}^{x=b} f(x)dx\approx \frac{h}{2}(f(x_0)+2f(x_1)+2f(x_2)+...+2f(x_{n-1})+f(x_n))$

        Error: $-\frac{1}{12}f''(\xi)(b-a)h^2$

        3. **Simpson's Rule**:
        **Simpson's Rule**: $\int_{x=a}^{x=b} f(x)dx\approx \frac{h}{3}[f(x_0)+4f(x_1)+f(x_2)]-\frac{h^5}{90}f^{(4)}(\xi)$

        **Composite Simpson's Rule**:$\int_{x=a}^{x=b} f(x)dx\approx \frac{h}{3}[f(x_0)+4f(x_1)+2f(x_2)+4f(x_3)+2f(x_4)+...+f(x_n)]$
        
        Error: $-\frac{h^4(b-a)}{180}f^{(4)}(\xi)$''')
        codes_int='''import scipy.integrate as integrate
import numpy as np
#integrate sin(x) from 0 to pi
def integrand(x):
  return np.sin(x)
area,error=integrate.quad(integrand,a=0,b=np.pi)
print("area under the curve:",area)
print("estimate of absolute error:",error)'''
        st.code(codes_int,language='python')


