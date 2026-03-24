# Autores: Ignacio Cañizares / Pedro Pérez / César Vidal

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Constantes
N_VALORES = [10, 20, 100, 200, 1000, 2000, 10000, 20000, 100000, 200000, 1000000] # Posibles tamaños del vector
LIMITE_TIEMPO_S = 60  # si tarda más de 60 s, indicar y detener

def construir_A_sparse(n):
    # Diagonales
    diag_principal = 4.0 * np.ones(n)
    diag_superior = -1.0 * np.ones(n - 1)
    diag_inferior = -1.0 * np.ones(n - 1)

    # Matriz tridiagonal
    A_sparse = sparse.diags(
        [diag_inferior, diag_principal, diag_superior],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="lil",
    )

    # Elementos cíclicos (esquinas)
    A_sparse[0, n - 1] = -1.0
    A_sparse[n - 1, 0] = -1.0

    # Convertir a CSC para spsolve
    return A_sparse.tocsc()




print(
    f"\n{'n':>10}  {'Tiempo (s)':>12}  {'Error máx':>14}  {'NNZ (no nulos)':>16}  {'Mem. sparse (KB)':>18}"
)
print("-" * 76)

for n in N_VALORES:
    t0 = time.perf_counter()
    A_sp = construir_A_sparse(n)
    b = np.ones(n)
    x = spsolve(A_sp, b)
    t1 = time.perf_counter()

    elapsed = t1 - t0

    x_exacta = np.full(n, 0.5)
    error = np.max(np.abs(x - x_exacta))

    nnz = A_sp.nnz
    # Memoria aproximada del formato CSC: datos + índices de fila + punteros de columna
    mem_sparse_kb = (nnz * 8 + nnz * 4 + (n + 1) * 4) / 1024.0

    print(
        f"{n:>10}  {elapsed:>12.6f}  {error:>14.2e}  {nnz:>16}  {mem_sparse_kb:>16.2f} KB"
    )

    if elapsed > LIMITE_TIEMPO_S:
        print(f"  → Tiempo excesivo ({elapsed:.1f} s). Se detiene para n mayores.")
        break


# print("\n" + "=" * 75)
# print("RESUMEN COMPARATIVO: Denso vs Jacobi Cíclico vs Sparse")
# print("=" * 75)
# print("""
# Método              | Memoria       | Tiempo por iter | Escala a n grande
# --------------------|---------------|-----------------|-------------------
# Denso (numpy)       | O(n^2)        | O(n^2) Jacobi   | No (>n~5000)
#                     |               | O(n^3) LU       |
# Jacobi cíclico      | O(n)          | O(n)            | Sí (n>200,000)
# scipy.sparse+spsolve| O(n) sparse   | O(n^1.5) LU sp. | Sí (n>1,000,000)
# """)
# print("""
# Conclusiones:
#   1. Para n pequeño (<200), los tres métodos son comparables.
#   2. A partir de n~1000, el enfoque denso se vuelve lento y costoso en RAM.
#   3. scipy.sparse es más rápido por iteración que Jacobi (LU vs iterativo),
#      pero requiere construir la matriz (aunque sparse).
#   4. jacobi_ciclico es la solución más ligera en memoria (2 vectores n).
#   5. En Stable Fluids real, se prefiere Jacobi cíclico o Gauss-Seidel
#      por su facilidad de paralelización en GPU (Taichi/CUDA).
# """)
