# Simulación de SFX — Laboratorio 1 (Grupal)

> **Grado en Ingeniería del Software**
> U-tad — Curso 25/26

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
