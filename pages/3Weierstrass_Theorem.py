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
    with st.expander("Bernstein Polynomials"):
        st.markdown("### Definition")
        st.markdown(r'''With f a real-valued function defined and bounded on the interval [0, 1], 
        let $B_n(f)$ be the polynomial on [0, 1] that
assigns to x the value $\sum_{k=0}^{n} nC_k x^k (1-x)^{n-k} f( \frac {k} {n})$.
$B_n(f)$ is the nth **Bernstein polynomial** for f. 

$B_n(f)$ converges uniformly to f, which means $B_n(f)(0)=f(0)$ and $B_n(f)(1)=f(1)$. Suppose Y= B(n; x) is the binomial random variable, which equals the
number of successes in n independent Bernoulli trials and where the probability of success on
each trial is x. 
The possible values of Y are 0,1,..,n and $P(Y=k)=nC_k x^k(1-x)^{n-k}$. 
For a given function f in C[0,1], let Z be
the random variable that assumes the value $f(\frac{k}{n})$ when Y=k. So Z assumes the value of $f(\frac{k}{n})$ with probability $nC_k x^k (1-x)^{n-k}$ and $B_n(x)$
is the expected value of Z. Since $\frac{n}{k}$ is the percentage of successes in n trials, it is
intuitively clear that $\frac{n}{k} \to x$ as $n \to \infty$. It is therefore easy
to conjecture that $(B_nf)(x)$, the expected value of Z, should approach f(x) as $n \to \infty$. 
''')
        st.markdown("### Some Identities")
        st.markdown(r'''
        $B_n(1)=\sum_{k=0}^{n} nC_k x^k(1-x)^{n-k}=1$

        $B_n(x)=\sum_{k=0}^{n} \frac{n}{k}nC_k x^k(1-x)^{n-k}=x$

        $B_n(x^2)=\sum_{k=0}^{n} \frac{n^2}{k^2} nC_k x^k(1-x)^{n-k}=\frac{(n-1)x^2+x}{n}$

        $B_n(x^3)=\sum_{k=0}^{n} \frac{n^3}{k^3} \frac{n}{k}nC_k x^k(1-x)^{n-k}$

        ...
    ''')
        st.markdown("### Proof of Weierstrass' Theorem")
        st.markdown(r'''
        $B_n(f)-f(x)=\sum_{k=0}^{n} nC_k x^k (1-x)^{n-k} f( \frac {k} {n})-f(x)\sum_{k=0}^{n} nC_k x^k (1-x)^{n-k}$
        
        $=\sum_{k=0}^{n} nC_k x^k (1-x)^{n-k}[f( \frac {k} {n})-f(x)]$
        
        It follows that $|B_n(f)-f(x)|\le \sum_{k=0}^{n} nC_k x^k (1-x)^{n-k} |f( \frac {k} {n})-f(x)|$ for each x on [0,1].
        
        To estimate this last sum, we separate the terms into two sums, $\Sigma'$ and $\Sigma''$, those for which $|\frac{k}{n}-x|<\delta$ and the remaining terms.
        Suppose that x is a point of continuity of f. For any $\epsilon>0$, there is a positive $\delta$ such that $|f(\frac{k}{n})-f(x)|<\frac{\epsilon}{2}$ when $|\frac{k}{n}-x|<\delta$. 
        
        For the first sum, $\Sigma'nC_k x^k (1-x)^{n-k} |f( \frac {k} {n})-f(x)|<\Sigma' C_k x^k (1-x)^{n-k} \frac{\epsilon}{2}$
        
       $ \le\sum_{k=0}^{n} nC_k x^k (1-x)^{n-k}\frac{\epsilon}{2}=\frac{\epsilon}{2}$.
       
       For the second sum, we have $ {\delta}^2 \le|\frac{k}{n}-x|^2$.  
       
       ${\delta}^2\Sigma''nC_k x^k (1-x)^{n-k} |f( \frac {k} {n})-f(x)|$
       
       $\le\Sigma''nC_k(\frac {k} {n}-x)^2 x^k (1-x)^{n-k} |f( \frac {k} {n})-f(x)|$
       
       $\le\Sigma''nC_k x^k (1-x)^{n-k} 2M=2M\frac{x(1-x)}{n}\le \frac{2M}{n}$
       
       Thus $\Sigma''nC_k x^k (1-x)^{n-k} |f( \frac {k} {n})-f(x)|\le \frac{2M}{{\delta}^2n}$.
       
       For this $\delta$, we can choose $n_0$ large enough so that $\frac{2M}{{\delta}^2n}\le \frac{\epsilon}{2} when n\ge n_0$. For such an n and the given x,
       $|B_n(f)-f(x)|\le \Sigma'+\Sigma''<\frac{\epsilon}{2}+\frac{\epsilon}{2}=\epsilon$. Hence, $B_nf(x)\to f(x)$ as $n\to \infty$ 
       for each point x of continuity of the function f.''')
