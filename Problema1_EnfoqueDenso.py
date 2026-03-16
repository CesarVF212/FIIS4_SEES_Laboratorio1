# AUTORES: Ignacio Cañizares / Pedro Pérez / César Vidal

# --------------------- #
# --- IMPORTACIONES --- #
# --------------------- #

import time
import numpy as np
from jacobi import jacobi

# ERORES de memoria de numpy.
from numpy._core._exceptions import _ArrayMemoryError


# ------------------ #
# --- CONSTANTES --- #
# ------------------ #

# Posibles valores del tamaño de la matriz cuadrada.
N_VALORES = [10, 20, 100, 200, 1000, 2000, 10000, 20000, 100000]

# Tolerancia permitida.s
TOL = 1e-6

# Número maximo de iteraciones permitidas.
MAXIT = 10000

# Límite de memoria de 16GB para no pasarse. Sin esto las últimas pruebas requieren hasta 60GBs de memoria.
LIMITE_MEMORIA = 2**(10) * 16


# ------------------------------------------ #
# --- FUNCIÓN: construcción de la matriz --- #
# ------------------------------------------ #

def construir_A(n):

    """
    Para construir esta matriz densa cíclica tridiagonal A de tamaño n x n, hemos seguido estos pasos:

    1. Rellenamos nuestra matriz A con ceros
    2. Usamos np.fill_diagonal para establecer la diagonal principal con el valor 4.
    3. Usamos la misma función pero en este caso para rellenar las diagonales adyacentes a la principal
    4. Añadimos las esquinas de forma manual.

    """

    A = np.zeros((n, n))

    # Diagonal principal.
    np.fill_diagonal(A, 4)

    # Rellenamos la subdiagona y la superdiagonal con valor -1.
    np.fill_diagonal(A[1:], -1)
    np.fill_diagonal(A[:, 1:], -1) 

    # Rellenamos las esquinas con -1.
    A[0, n - 1] = -1
    A[n - 1, 0] = -1

    return A


# ------------------------ #
# --- CÓDIGO PRINCIPAL --- #
# ------------------------ #


# --- PASO 1: Construcción de A y medición de tiempo/memoria para cada n.

print("\n\n--- PASO 1: Construcción de A y medición de tiempo/memoria\n")
print(f"{'n':>8} {'Tiempo (s)':>12} {'Memoria A ':>16}")
print("-" * 42)

matrices = [None] * len(N_VALORES)

# ITERAMOS TODOS LOS VALORES N. # Introducimos un contador para ir almacenando las matrices.
for i, n in enumerate(N_VALORES):

    # Estimamos el tamaño en memoria de la matriz. Cada valor ocupa un byte.
    memoria_estimada = (n**2) / (1024**2) 

    # Indicamos el caso de estar pasandonos en memoria.
    if memoria_estimada > LIMITE_MEMORIA:
        print(
            f"{n:>8} {'—':>12} {memoria_estimada / 2**(10 ):>14.1f} GB (omitido: excede límite)"
        )
        continue

    # Iniciamos para cada matriz su contador de tiempo.
    t0 = time.perf_counter()
    A = construir_A(n)
    b = np.ones(n)
    t1 = time.perf_counter()

    # Guardamos la matriz en el array:
    matrices[i] = A

    # Medimos la memoria real una vez creada.
    memoria_real = A.nbytes / (1024**2)
    
    # En el caso de ocupar mas de 1GB cambiamos el formato de visualización.
    if(memoria_real < 2**(10)):
        print(f"{n:>8} {t1 - t0:>12.6f} {memoria_real:>14.4f} MB")

    else:
        print(f"{n:>8} {t1 - t0:>12.6f} {memoria_real / 2**10 :>14.4f} GB")


# --- PASO 2: Resolución mediante Jacobi.

print("\n\n--- PASO 2: Resolución con Jacobi\n")
print(f"{'n':>8} {'Tiempo (s)':>12} {'Error máx':>14} {'Iters':>8}")
print("-" * 42)

for i, n in enumerate(N_VALORES):

    if matrices[i] is not None:
        
        A = matrices[i]

        # Los términos independientes son siempre uno.
        b = np.ones(n)

        # Las soluciones se obtienen en base float32.
        x0 = np.zeros(n)

        t0 = time.perf_counter()
        x_jac, iters = jacobi(A, b, x0, tol=TOL, maxit=MAXIT, verbose=False)
        t1 = time.perf_counter()

        # Calculamos el error sabiendo que la solución exacta es 0.5 para todos los casos.
        x_exacta = np.full(n, 0.5)
        error = np.max(np.abs(x_jac - x_exacta))

        print(f"{n:>8} {t1 - t0:>12.6f} {error:>14.2e} {iters:>8}")

    else:
        print(f"La matriz A para el valor N = {n} no fue creada por falta de memoria o un error.")


## --- PASO 3: Resolución mediante Sistemas Lineales.

print("\n\n--- PASO 3: Resolución con numpy.linalg.solve:\n")
print(f"{'n':>8} {'Tiempo (s)':>12} {'Error máx':>14}")
print("-" * 42)

for i, n in enumerate(N_VALORES):

    if matrices[i] is not None:
        
        A = matrices[i]
        b = np.ones(n, dtype=np.bool)

        try:
            t0 = time.perf_counter()
            x_sol = np.linalg.solve(A, b)
            t1 = time.perf_counter()

            x_exacta = np.full(n, 0.5)
            error = np.max(np.abs(x_sol - x_exacta))

            print(f"{n:>8} {t1 - t0:>12.6f} {error:>14.2e}")

        except (_ArrayMemoryError, ValueError, MemoryError) as e:
            print(f"ERROR: Sin memoria para n={n}: {e}")

    else:
        print(f"La matriz A para el valor N = {n} no fue creada por falta de memoria o un error.")

        
print("\nNota: numpy.linalg.solve usa factorización LU (O(n^3)), más rápido")
print("para n pequeño pero inviable para n muy grande por memoria O(n^2).")
print("Es inviable usar esta función de numpy por usa por defecto float64, que hace que alocar la cantidad de memoria necesaria sea inviable.")