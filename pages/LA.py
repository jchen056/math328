import streamlit as st
import numpy as np
import pandas as pd
import re
from scipy.linalg import lu

st.sidebar.markdown("### Linear algebra")

tb1,tb2,tb3=st.tabs(['Norms','Solve Ax=b','Other'])
 
with tb1:
    st.sidebar.markdown("#### Norms")
    st.markdown(r'''### vector norm
1. **L2 norm**(Euclian length): $||v||_2=\sqrt{\sum_{i=1}^{n}(v_i)^2}$
1. **L1 norm**(Manhattan norm): $||v||_1=\sum_{i=1}^{n}|v_i|$
1. **Lp norm**(Minkowski norm): $||v||_p=\sqrt[p]{(\sum_{i=1}^{n}(v_i)^p)}$
1. **L-$\infty$ norm**(sup-norm): $|v|_{\infty}=\max_{i} |v_i|$ 
### matrix norm
* A is m by n matrix
* $||A||_2=\sqrt{\rho (A^TA)}$  where $\rho (B)$ is the largest eigenvalue of B in absolute value
* $||A||_1=max_{1 \le j \le n} \sum_{i=1}^{m}|a_{ij}|$
* $||A||_p=\sqrt[p]{\sum_{i}^{m}\sum_{j}^{n}|a_{ij}|^p}$ where A is a mXn matrix
* $||A||_{\infty}=max_{1 \le i \le m} \sum_{j=1}^{n}|a_{ij}|$

''')
    with st.expander("Norm Calculator"):
        column1,column2=st.columns(2)
        with column1:
            st.markdown("Vector norm")
            input1=st.text_input("Enter your vector(separated by commas)")
            x=list(map(int,re.findall(r'\d+', input1)))
            option = st.selectbox('What type of norm you want to calculate?',
    ('inf', 1, 2,'-inf'))
            if st.button("click here to compute the vector norm"):
                if option==1 or option==2:
                    st.write(np.linalg.norm(x,option))
                elif option=='inf':
                    st.write(np.linalg.norm(x,np.inf))
                elif option=='-inf':
                    st.write(np.linalg.norm(x,-np.inf))


        with column2:
            st.markdown("Matrix norm")
            inp2=st.text_area("Enter your matrix(separate elements by commas and rows by entering a new line)")
            txt_ls=inp2.splitlines()
            A=[]
            for i in txt_ls:
                rows=list(map(int,re.findall(r'\d+',i)))
                if len(rows)!=0:
                    A.append(rows)
            opt=st.selectbox("What matrix norm you want to compute?",('inf', 1, 2,'-inf','Frobenius'))
            if st.button("click here to compute the vector norm",key=2):
                if opt==1 or opt==2:
                    matrix_norm=np.linalg.norm(A,opt)
                elif opt=="inf":
                    matrix_norm=np.linalg.norm(A,np.inf)
                elif opt=='-inf':
                    matrix_norm=np.linalg.norm(A,-np.inf)
                elif opt=='Frobenius':
                    matrix_norm=np.linalg.norm(A,'fro')
                st.write(matrix_norm)
            
with tb2:
    st.markdown('''There are many ways to solve systems of linear equations $Ax=b$. In general, there are two methods--**direct methods** and **iterative methods**.''')
    with st.expander("Direct methods"):
        st.markdown('''### LU decomposition''')
        st.markdown('''
        Solving Ax=b using LU decomposition has three steps:
        * Gaussian elimination: A=LU => LUx=b
        * Forward substituion: Ux=c, solve for c in Lc=b
        * Backward substitution: solve x in ux=c''')
        subt1,subt2,subt3=st.tabs(['Building codes from scratch','Numpy codes','error'])
        with subt1:
            codes_LU='''import numpy as np
    def LU_dec(A):#The elements of A has dtype of float
        m,n=A.shape
        if m!=n:
            print("A is not a square; LU only applies to square matrix")
            return
        else:
            L=np.identity(n)
            for j in range(n-1):#loop each column from j=0 to j=n-2
                for i in range(j+1,n):#loop rows
                    L[i,j]=(A[j+1,j]/A[j,j])#multiplier
                    A[i,:]=A[i,:]-L[i,j]*A[j,:]
            return L,A#return L,U'''
            st.code(codes_LU,language='python')
            codes_forward='''def forward_sub(L,b):
        n=len(L)
        
        b[0]=b[0]/L[0,0]
        for i in range(1,n):#row values
            for j in range(i):
                b[i]=b[i]-L[i,j]*b[i-1]
            b[i]=b[i]/L[i,i]
        return b#our c'''
            st.code(codes_forward,language='python')
            codes_backward='''def backward_sub(U,c):
        n=len(U)
        x=np.zeros(n)
        for i in range(n-1,-1,-1):
            x[i]=c[i]
            for j in range(i+1,n):
                x[i]-=U[i,j]*x[j]
            x[i]=x[i]/U[i,i]
        return x'''
            st.code(codes_backward,language='python')
        with subt2:
            codes_lpsolve='''A = np.array([[4, 3, -5], 
              [-2, -4, 5], 
              [8, 8, 0]])
b = np.array([2, 5, -3])
x = np.linalg.solve(A, b)
print(x)'''
            st.code(codes_lpsolve,language='python')
            codes_lu='''from scipy.linalg import lu
P, L, U = lu(A)
print('P:', P)
print('L:', L)
print('U:', U)
print('LU:',np.dot(L, U))'''
            st.code(codes_lu,language='python')
            st.markdown('''If the matrix A is **symmetric** and **positive definite**, A can be factored as $A=GG^T$ where $G=LD^{1/2}$. (**Cholesky factorization**)''')
        with subt3:
            st.markdown(r'''By some algorithm, we can compute an approximate solution $\overline{x}$ to Ax=b. We are concerned with the following error measurements:''')
            st.latex(r'''absolute error: ||x-\overline{x}||''')
            st.latex(r'''relative error: \frac{||x-\overline{x}||}{||x||}''')
            st.latex(r'''residual: \overline{r}=b-A\overline{x}''')
            st.markdown(r'''
            Residual can be easily computed, where absolute error and relative error are impossible to compute.
            * $\overline{r}=b-A\overline{x}=Ax-A\overline{x}$
            * $x-\overline{x}=A^{-1}\overline{r}$
            * $||e||=||x-\overline{x}||\le ||A^{-1}||||\overline{r}||$
            * $||A||=||A^{-1}b||\le \frac{||b||}{||A||}$
            * $\frac{||x-\overline{x}||}{||x||} \le ||A^{-1}||||\overline{r}||\frac{||A||}{||b||}$
            * **condition number** of square matrix A is $k(A)=||A|||||A^{-1}||$
            * $\frac{||x-\overline{x}||}{||x||} \le k(A) \frac{||\overline{r}||}{||b||}$
            
            The condition number measures the sensitivity of a problem. A large condition number means the problem is ill-conditioned.
            ''')
            st.markdown("**Condition number calcualtor**")
            inp2=st.text_area("Enter your matrix(separate elements by commas and rows by entering a new line)",key=4)
            txt_ls=inp2.splitlines()
            A=[]
            for i in txt_ls:
                rows=list(map(float,re.findall(r'-?\d+',i)))
                if len(rows)!=0:
                    A.append(rows) 
       
            opt=st.selectbox("What type of condition number you want to compute?",('inf', 1, 2,'-inf','Frobenius'),key=3)
            if st.button("click here to compute condition number"):
                if opt==1 or opt==2:
                    st.write(np.linalg.cond(A,opt))
                elif opt=='inf':
                    st.write(np.linalg.cond(A,np.inf))
                elif opt=='Frobenius':
                    st.write(np.linalg.cond(A,'fro'))
                elif opt=='-inf':
                    st.write(np.linalg.cond(A,-np.inf))
    with st.expander("Least squares"):
        st.markdown('''Instead of solving Ax=b, we will solve $\min_{x}||b-Ax||_2$ where A has full column rank.
        Instances of least squares arise in machine learning, computer vision, and computer graphics applications.''')
        st.markdown('''
        Here are standard methods for solving the linear least squares problems:
        * **Normal equations**: fast, simple, intuitive, but less robust in ill-conditioned situtions(large k(A)).
        * **QR decomposition**: used in general-purpose software. More computational expensive than normal equations but more robust.
        * **SVD**: very robust but significantly more expensive. ''')
        tb_m1,tb_m2=st.tabs(['Normal equations','QR decomposition'])
        with tb_m1:
            st.markdown(r'''
            ### Normal Equations
            * $\overline{x}=\min_{x}||b-Ax||_2$
            * residual $r=b-A\overline{x}$ is perpendicular to A $\implies A^T(b-Ax)=0 \implies A^TAx=A^Tb$ 
            * Normal equation: $A^TAx=A^Tb$
            * Least square solution: $\overline{x}=(A^TA)^{-1}A^Tb$
            * Projection: Pb=$A\overline{x}=A(A^TA)^{-1}A^Tb$''')
            codes_n='''def normal_eqn2(A,b):
    B=A.T@A
    y=A.T@b
    
    #B is symmetric, pos def, where Cholesky factorization comes in
    G=np.linalg.cholesky(B)
    
    #Bx=y => G G.T x=y; using forward substitution and backward substitution
    z=np.linalg.solve(G,y)
    x=np.linalg.solve(G.T,z)
    return x'''
            st.code(codes_n,language='python')
            codes_n1='''def normal_equations1(A,b):
    x=np.linalg.lstsq(A,b,rcond=None)
    return x[0]'''
            st.code(codes_n1,language='python')
            st.markdown('''The main drawback of the normal equations for solving the least square problems is accuracy in the presence of large condition numbers.
            ''')
        with tb_m2:
            st.markdown(r'''
            ### Orthogonal Transformation and QR
            * A=QR where Q is orthogonal($Q^TQ=I$)
            * $Ax=b \implies QRx=b \implies Q^TQRx=Q^Tb \implies Rx=Q^Tb$
            * To factor A into QR, we can use **Gram-Schmidt** or **Householder reflector**''')
            codes_qr='''def QR_fac(A,b):
    Q,R=np.linalg.qr(A)
    x=np.linalg.solve(R,Q.T@b)
    return x'''
            st.code(codes_qr,language='python')
            
            

