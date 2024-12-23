# **Regression Algorithms Implementations from Scratch with Python**

`regression_algorithms.py` implements four popular regression algorithms **using mathematical equations** and **minimizing the imports of existing libraries**.

---

## **Implemented Regression Algorithms**

1. **Least-Squares (LS)**  
2. **Regularized Least-Squares (RLS)** = **L2-regularized LS** (Ridge Regression)  
3. **L1-regularized LS (LASSO)**  
4. **Robust Regression (RR)**  

---

## **Mathematical Equations**

### 1. **Least-Squares (LS)**  
The **least-squares** problem minimizes the sum of squared residuals:

\[
\min_{\theta} \| y - \Phi \theta \|_2^2
\]

The solution is derived using the **normal equations**:

\[
\theta = (\Phi^T \Phi)^{-1} \Phi^T y
\]

Where:  
- \(\Phi\) is the **design matrix** (shape: \(m \times n\))  
- \(y\) is the **target vector** (shape: \(m\))  
- \(\theta\) is the **parameter vector** (shape: \(n\))  

---

### 2. **Regularized Least-Squares (RLS)** = Ridge Regression  
The **Ridge Regression** adds an L2-regularization term to the least-squares objective:

\[
\min_{\theta} \| y - \Phi \theta \|_2^2 + \lambda \|\theta\|_2^2
\]

The solution is derived as:

\[
\theta = (\Phi^T \Phi + \lambda I)^{-1} \Phi^T y
\]

Where:  
- \(\lambda\) is the **regularization parameter** that controls the trade-off between bias and variance.  
- \(I\) is the identity matrix of size \(n \times n\).  

---

### 3. **L1-Regularized Least-Squares (LASSO)**  
The **LASSO** regression minimizes the sum of squared residuals with an L1-regularization term:

\[
\min_{\theta} \| y - \Phi \theta \|_2^2 + \lambda \|\theta\|_1
\]

This problem is solved using **quadratic programming**. The optimization problem is formulated as:

\[
\text{Minimize: } \frac{1}{2} \theta^T Q \theta + c^T \theta
\]

Subject to:  
\[
G \theta \leq h
\]

Where:  
- \(Q = \Phi^T \Phi\),  
- \(c = \lambda \mathbf{1} - \Phi^T y\),  
- \(G\) and \(h\) are matrices that enforce L1 constraints.  

---

### 4. **Robust Regression (RR)**  
The **Robust Regression** minimizes the **L1-norm** of residuals, making it less sensitive to outliers:

\[
\min_{\theta} \| y - \Phi \theta \|_1
\]

This is formulated as a **linear programming** problem:

\[
\text{Minimize: } \mathbf{1}^T z
\]

Subject to:  
\[
\Phi \theta - z \leq y, \quad -\Phi \theta - z \leq -y
\]

Where:  
- \(z\) represents the absolute values of the residuals.  
- Linear programming ensures that the L1-norm is minimized.
