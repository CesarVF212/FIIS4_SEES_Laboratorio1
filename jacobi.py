# AUTORES: Ignacio Cañizares / Pedro Pérez / César Vidal

# --------------------- #
# --- IMPORTACIONES --- #
# --------------------- #

import numpy as np


# ------------------------------------ #
# --- FUNCIÓN: Método de Jacobi    --- #
# ------------------------------------ #

def jacobi(A, b, x0, tol=1.0e-6, maxit=200, verbose=True):

    if verbose:
        print("k\txk\t\t\terror")
        print(f"0\t{x0}\t")

    xk = x0

    # --- Bucle iterativo principal.

    for k in range(1, maxit + 1):

        # Guardamos la iteración anterior.
        xkprev = xk.copy()

        # Fórmula de Jacobi: x(k) = x(k-1) + (b - A·x(k-1)) / diag(A)
        xk = xkprev + (b - A @ xkprev) / np.diag(A)

        # Calculamos el error como la norma infinito de la diferencia.
        error = np.linalg.norm(xk - xkprev, np.inf)

        if verbose:
            print(
                f"{k}\t{np.array2string(xk, formatter={'float': lambda x: f'{x:.6f}'})}\t{error:e}"
            )

        # Si el error es menor que la tolerancia, hemos convergido.
        if error < tol:
            break

    else:
        
        # Se ejecuta si el for termina sin break (no convergió).
        print(f"Número máximo de iteraciones {maxit} alcanzado.")
        xk = None

    return xk, k