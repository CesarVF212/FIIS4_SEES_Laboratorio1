# --- ENGLISH ---

# SFX Simulation — Lab 1 (Group)

This lab solves a cyclic tridiagonal linear system **Ax = b** for increasing sizes *n*, comparing a dense approach (Jacobi and `numpy.linalg.solve`) with a matrix-free cyclic Jacobi implementation.
It measures time, memory and error for each method, and optionally compares them with `scipy.sparse`.

> **BSc in Software Engineering**
> U-tad — Academic Year 25/26
> Pedro Pérez, Ignacio Cañizares and César Vidal.

---

## § Instructions §

- The lab code must be organized in a Python script (`.py`) for each problem, properly documented.
- A presentation explaining the development of the work, results and conclusions must be included. Optionally, a video presentation can be included.
- The use of code from references or language models (LLMs) such as ChatGPT or GitHub Copilot is **not allowed**.
- All files must be submitted compressed in a `.zip` file.

---

## Jacobi Method for Sparse Systems

Consider the following *cyclic tridiagonal matrix* **A** of size *n × n* and the vector of ones **b** of size *n*:

$$
\mathbf{A} = \begin{pmatrix}
4 & -1 & 0 & \cdots & 0 & -1 \\
-1 & 4 & -1 & \cdots & 0 & 0 \\
0 & -1 & 4 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 4 & -1 \\
-1 & 0 & 0 & \cdots & -1 & 4
\end{pmatrix}, \qquad
\mathbf{b} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ \vdots \\ 1 \\ 1 \end{pmatrix}
$$

The goal is to solve the system **Ax = b**. The exact solution is **x** = (1/2, 1/2, …, 1/2) for any value of *n*.

> [!NOTE]
> Each problem must be solved for increasing values of *n* = 10, 20, 100, 200, 1000, 2000, 10000, 20000, …
> as far as possible or reasonable. It is not necessary to reach the same value of *n* in all parts:
> if at some point the program runs out of memory or takes too long, it is enough to state it.

---

## Problem 1: Dense Approach

### (1) Construction of the dense matrix

Implement a function `construir_A(n)` that returns the matrix **A** as an `np.array` of size *n × n*.

> *Note: `np.diag` or `np.fill_diagonal` can be used to build the diagonals.*

- For each value of *n*, build the matrix **A** and the vector **b**, measuring the construction time with `time.perf_counter()` and the memory occupied by **A** with `A.nbytes`.
- **How much memory (in MB or GB) would the dense matrix A need for *n* = 100,000?**

### (2) Solving with Jacobi

Use the `jacobi` function available in `jacobi.py` to solve the system, measuring the time. Compute the error with respect to the exact solution.

> *Note: take **x₀ = 0** as the initial vector.*

### (3) Solving with `numpy.linalg.solve`

Solve the system with `numpy.linalg.solve` for each value of *n*, measuring the time. Compute the error with respect to the exact solution and compare with the times from the previous part.

---

## Problem 2: Jacobi for the Cyclic System

For this particular system, the equations of the Jacobi method can be written without referring to the matrix **A**:

$$
x_0^{(k)} = \frac{1}{4}\left(1 + x_1^{(k-1)} + x_{n-1}^{(k-1)}\right)
$$

$$
x_i^{(k)} = \frac{1}{4}\left(1 + x_{i-1}^{(k-1)} + x_{i+1}^{(k-1)}\right), \quad i = 1, \ldots, n-2
$$

$$
x_{n-1}^{(k)} = \frac{1}{4}\left(1 + x_0^{(k-1)} + x_{n-2}^{(k-1)}\right)
$$

### (1) Implementation of `jacobi_ciclico`

Implement a function `jacobi_ciclico(n, tol, maxit)` that solves the system directly using these equations, **without building or storing the matrix A**. The only input parameters must be `n`, the tolerance and the maximum number of iterations.

### (2) Solving and measuring

For each value of *n*, solve the system with `jacobi_ciclico`, measuring the time and computing the error with respect to the exact solution. Also state the memory used (only 2 vectors of size *n* are needed).

### (3) Comparison

Compare times and memory with the results of Problem 1. **What conclusions can be drawn?**

---

## Problem 3: `scipy.sparse` *(Optional)*

The `scipy.sparse` module allows sparse matrices to be stored efficiently, keeping only the non-zero elements. The following code builds the matrix **A** in sparse format and solves the system:

```python
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Diagonals
diag_principal = 4.0 * np.ones(n)
diag_superior  = -1.0 * np.ones(n - 1)
diag_inferior  = -1.0 * np.ones(n - 1)

# Tridiagonal matrix
A_sparse = sparse.diags(
    [diag_inferior, diag_principal, diag_superior],
    offsets=[-1, 0, 1],
    shape=(n, n),
    format="lil"
)

# Cyclic elements (corners)
A_sparse[0, n - 1] = -1.0
A_sparse[n - 1, 0] = -1.0
A_sparse = A_sparse.tocsc()

b = np.ones(n)
x = spsolve(A_sparse, b)
```

### (1) Solving and comparison

Use this code to solve the system for the same values of *n*, measuring times. Compare with the results of Problems 1 and 2.

---

> **Deadline:** March 25, 2026 at 23:59

# --- SPANISH ---

# Simulación de SFX — Laboratorio 1 (Grupal)

> **Grado en Ingeniería del Software**
> U-tad — Curso 25/26
> Pedro Pérez, Ignacio Cañizares y César Vidal.

---

## § Instrucciones §

- El código del laboratorio debe organizarse en un script de Python (`.py`) por cada problema, debidamente documentado.
- Se debe incluir una presentación que explique el desarrollo del trabajo, resultados y conclusiones. Opcionalmente, se puede incluir una video-presentación.
- **No está permitido** el uso de código procedente de referencias o modelos de lenguaje (LLMs) como ChatGPT o Github Copilot.
- Todos los ficheros deben entregarse comprimidos en un fichero `.zip`.

---

## Método de Jacobi para sistemas sparse

Consideremos la siguiente _matriz cíclica tridiagonal_ **A** de tamaño _n × n_ y el vector de unos **b** de tamaño _n_:

$$
\mathbf{A} = \begin{pmatrix}
4 & -1 & 0 & \cdots & 0 & -1 \\
-1 & 4 & -1 & \cdots & 0 & 0 \\
0 & -1 & 4 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 4 & -1 \\
-1 & 0 & 0 & \cdots & -1 & 4
\end{pmatrix}, \qquad
\mathbf{b} = \begin{pmatrix} 1 \\ 1 \\ 1 \\ \vdots \\ 1 \\ 1 \end{pmatrix}
$$

El objetivo es resolver el sistema **Ax = b**. La solución exacta es **x** = (1/2, 1/2, …, 1/2) para cualquier valor de _n_.

> [!NOTE]
> Cada problema debe resolverse para valores crecientes de _n_ = 10, 20, 100, 200, 1000, 2000, 10000, 20000, …
> hasta donde sea posible o razonable. No es necesario llegar al mismo valor de _n_ en todos los apartados:
> si en algún momento el programa se queda sin memoria o tarda demasiado, basta indicarlo.

---

## Problema 1: Enfoque denso

### (1) Construcción de la matriz densa

Implementar una función `construir_A(n)` que devuelva la matriz **A** como un `np.array` de tamaño _n × n_.

> _Nota: se puede utilizar `np.diag` o `np.fill_diagonal` para construir las diagonales._

- Para cada valor de _n_, construir la matriz **A** y el vector **b**, midiendo el tiempo de construcción con `time.perf_counter()` y la memoria ocupada por **A** con `A.nbytes`.
- **¿Cuánta memoria (en MB o GB) necesitaría la matriz A densa para _n_ = 100 000?**

### (2) Resolución con Jacobi

Utilizar la función `jacobi` disponible en `jacobi.py` para resolver el sistema, midiendo el tiempo. Calcular el error respecto a la solución exacta.

> _Nota: tomar **x₀ = 0** como vector inicial._

### (3) Resolución con `numpy.linalg.solve`

Resolver el sistema con `numpy.linalg.solve` para cada valor de _n_, midiendo el tiempo. Calcular el error respecto a la solución exacta y comparar con los tiempos del apartado anterior.

---

## Problema 2: Jacobi para el sistema cíclico

Para este sistema concreto, las ecuaciones del método de Jacobi se pueden escribir sin hacer referencia a la matriz **A**:

$$
x_0^{(k)} = \frac{1}{4}\left(1 + x_1^{(k-1)} + x_{n-1}^{(k-1)}\right)
$$

$$
x_i^{(k)} = \frac{1}{4}\left(1 + x_{i-1}^{(k-1)} + x_{i+1}^{(k-1)}\right), \quad i = 1, \ldots, n-2
$$

$$
x_{n-1}^{(k)} = \frac{1}{4}\left(1 + x_0^{(k-1)} + x_{n-2}^{(k-1)}\right)
$$

### (1) Implementación de `jacobi_ciclico`

Implementar una función `jacobi_ciclico(n, tol, maxit)` que resuelva el sistema utilizando directamente estas ecuaciones, **sin construir ni almacenar la matriz A**. Los únicos parámetros de entrada deben ser `n`, la tolerancia y el número máximo de iteraciones.

### (2) Resolución y medición

Para cada valor de _n_, resolver el sistema con `jacobi_ciclico`, midiendo el tiempo y calculando el error respecto a la solución exacta. Indicar además la memoria utilizada (solo se necesitan 2 vectores de tamaño _n_).

### (3) Comparación

Comparar tiempos y memoria con los resultados del Problema 1. **¿Qué conclusiones se extraen?**

---

## Problema 3: `scipy.sparse` _(Opcional)_

El módulo `scipy.sparse` permite almacenar matrices sparse de forma eficiente, guardando solo los elementos no nulos. El siguiente código construye la matriz **A** en formato sparse y resuelve el sistema:

```python
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Diagonales
diag_principal = 4.0 * np.ones(n)
diag_superior  = -1.0 * np.ones(n - 1)
diag_inferior  = -1.0 * np.ones(n - 1)

# Matriz tridiagonal
A_sparse = sparse.diags(
    [diag_inferior, diag_principal, diag_superior],
    offsets=[-1, 0, 1],
    shape=(n, n),
    format="lil"
)

# Elementos cíclicos (esquinas)
A_sparse[0, n - 1] = -1.0
A_sparse[n - 1, 0] = -1.0
A_sparse = A_sparse.tocsc()

b = np.ones(n)
x = spsolve(A_sparse, b)
```

### (1) Resolución y comparación

Utilizar este código para resolver el sistema para los mismos valores de _n_, midiendo tiempos. Comparar con los resultados de los Problemas 1 y 2.

---

> **Fecha de entrega:** 25 de marzo de 2026 a las 23:59
