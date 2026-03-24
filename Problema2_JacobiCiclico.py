# Autores: Ignacio Cañizares / Pedro Pérez / César Vidal

import time
import numpy as np
import taichi as ti

# Inicializar Taichi en gou nvidia (cuda)
ti.init(arch=ti.cuda, default_fp=ti.f64)

MAX_N = 2_000_000  # tamaño máximo soportado

# Campos en GPU (Taichi fields)
x_prev = ti.field(dtype=ti.f64, shape=MAX_N)  # iteración k-1
x_new = ti.field(dtype=ti.f64, shape=MAX_N)  # iteración k
diff_field = ti.field(dtype=ti.f64, shape=())  # acumulador escalar para diff
err_field = ti.field(dtype=ti.f64, shape=())  # acumulador escalar para error

# Constantes
N_VALORES = [
    10,
    20,
    100,
    200,
    1000,
    2000,
    10000,
    20000,
    100000,
    200000,
    1000000,
    2000000,
]
TOL = 1e-6
MAXIT = 10000

# Kernels
@ti.kernel
def init_cero(n: int):
    for i in range(n):
        x_prev[i] = 0.0


# En este kernel se calcula la nueva a partir de la previa. Además se paraleliza el bucle for
@ti.kernel
def jacobi_paso(n: int):
    for i in range(n):
        izq = x_prev[(i - 1 + n) % n]
        der = x_prev[(i + 1) % n]
        x_new[i] = (1.0 + izq + der) / 4.0


@ti.kernel
def copiar_new_a_prev(n: int):
    for i in range(n):
        x_prev[i] = x_new[i]


# Resultado escrito en diff_field[None] (campo escalar en gpu).
@ti.kernel
def calcular_diff(n: int):
    diff_field[None] = 0.0
    for i in range(n):
        d = ti.abs(x_new[i] - x_prev[i])
        ti.atomic_max(diff_field[None], d)


# El resultado exacto es x = 0.5 (viene en el encunciado), por lo que calculamos el error máximo respecto a esa solución.
@ti.kernel
def calcular_error(n: int):
    err_field[None] = 0.0
    for i in range(n):
        ti.atomic_max(err_field[None], ti.abs(x_prev[i] - 0.5))


def jacobi_ciclico(n, tol=1e-6, maxit=10000):
    if n > MAX_N:
        print(f"n={n} supera MAX_N={MAX_N}")
        return None, 0

    init_cero(n)                      # Inicializar x=0 en GPU

    for k in range(maxit):
        jacobi_paso(n)                # Nuevo x_new en paralelo
        calcular_diff(n)              # Diff antes de actualizar
        diff = diff_field[None]



        if diff < tol:                # Convergencia
            copiar_new_a_prev(n)
            calcular_error(n)
            return err_field[None], k + 1

        copiar_new_a_prev(n)          # x_prev <- x_new

    calcular_error(n)
    return err_field[None], maxit




# Primera llamada compila los kernels en CUDA
print("\nCompilando kernels cuda")
jacobi_ciclico(10, tol=TOL, maxit=MAXIT)
print("Compilación completada.\n")

print(
    f"{'n':>10}  {'Tiempo (s)':>12}  {'Error máx':>12}  {'Iters':>8}  {'Mem. 2 vecs (KB)':>18}"
)
print("-" * 70)

for n in N_VALORES:
    if n > MAX_N:
        print(f"{n:>10}  —  (supera MAX_N={MAX_N:,})")
        continue

    # Sincronizar gpu antes de medir para evitar tiempos de compilación
    ti.sync()
    t0 = time.perf_counter()
    error, iters = jacobi_ciclico(n, tol=TOL, maxit=MAXIT)
    ti.sync()  # esperar a que la gpu termine antes de parar el reloj
    t1 = time.perf_counter()

    mem_kb = (2 * n * 8) / 1024.0 # Memoria: 2 vectores float64
    print(f"{n:>10}  {t1 - t0:>12.6f}  {error:>12.2e}  {iters:>8}  {mem_kb:>16.2f} KB")
