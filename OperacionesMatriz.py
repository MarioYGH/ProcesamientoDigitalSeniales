import numpy as np

# Definir la matriz A
A = np.array([[2, -1, 0], 
              [1, 3, 1], 
              [0, -1, 2]])

# Vectores de prueba
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

c = 3  # Escalar de prueba

# Cálculo del determinante
det_A = np.linalg.det(A)

# Cálculo de la traza
trace_A = np.trace(A)

# Cálculo de los eigenvalores y eigenvectores
eigenvalues, eigenvectors = np.linalg.eig(A)

# Verificar si la matriz es invertible
if det_A != 0:
    A_inv = np.linalg.inv(A)
    invertible = True
else:
    A_inv = None
    invertible = False

# Aplicar la transformación T(x) = Ax
T_v1 = A @ v1
T_v2 = A @ v2
T_v1_v2 = A @ (v1 + v2)
T_c_v1 = A @ (c * v1)

# Verificar aditividad: T(v1 + v2) == T(v1) + T(v2)
additivity_check = np.allclose(T_v1_v2, T_v1 + T_v2)

# Verificar homogeneidad: T(c * v1) == c * T(v1)
homogeneity_check = np.allclose(T_c_v1, c * T_v1)

# Mostrar resultados con formato bonito
print("\n===== Matriz A y Propiedades =====")
print("Matriz A:\n", A)
print(f"Determinante de A: {det_A:.4f}")
print(f"Traza de A: {trace_A:.4f}")
print("\nEigenvalores de A:", eigenvalues)
print("Eigenvectores de A:\n", eigenvectors)
print("\n¿La matriz A tiene inversa?:", "Sí" if invertible else "No")

if invertible:
    print("\nInversa de A:\n", A_inv)

print("\n===== Transformaciones Lineales =====")
print(f"T(v1) = {T_v1}")
print(f"T(v2) = {T_v2}")
print(f"T(v1 + v2) = {T_v1_v2}")
print(f"T(v1) + T(v2) = {T_v1 + T_v2}")
print(f"Aditividad verificada: {'Sí' if additivity_check else 'No'}")

print(f"\nT(c * v1) = {T_c_v1}")
print(f"c * T(v1) = {c * T_v1}")
print(f"Homogeneidad verificada: {'Sí' if homogeneity_check else 'No'}")

# Verificación final
if additivity_check and homogeneity_check:
    print("\n La transformación es lineal.")
else:
    print("\n La transformación NO es lineal.")
