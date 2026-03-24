import numpy as np


# Por como habíamos construido la llamada en EnfoqueDenso.py, la función jacobi debe tener esta forma en la que también devuelve k:
def jacobi(A, b, x0, tol=1.0e-6, maxit=200, verbose=True):
    if verbose:
        print("k\txk\t\t\terror")
        print(f"0\t{x0}\t")

    xk = x0
    for k in range(1, maxit + 1):
        xkprev = xk.copy()
        xk = xkprev + (b - A @ xkprev) / np.diag(A)
        error = np.linalg.norm(xk - xkprev, np.inf)
        if verbose:
            print(
                f"{k}\t{np.array2string(xk, formatter={'float': lambda x: f'{x:.6f}'})}\t{error:e}"
            )
        if error < tol:
            break
    else:
        print(f"Número máximo de iteraciones {maxit} alcanzado.")
        xk = None

    return xk, k  # devolver también k


if __name__ == "__main__":
    import numpy as np

    A = np.array([[10, 1], [1, 8]])
    b = np.array([23, 26])
    x0 = np.zeros(len(b))
    jacobi(A, b, x0)
