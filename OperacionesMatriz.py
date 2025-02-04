import numpy as np

# Definir la matriz A (puedes cambiar los valores para probar otros casos)
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

# Mostrar resultados
print("T(v1) =", T_v1)
print("T(v2) =", T_v2)
print("T(v1 + v2) =", T_v1_v2)
print("T(v1) + T(v2) =", T_v1 + T_v2)
print("Aditividad verificada:", additivity_check)

print("\nT(c * v1) =", T_c_v1)
print("c * T(v1) =", c * T_v1)
print("Homogeneidad verificada:", homogeneity_check)

# Verificación final
if additivity_check and homogeneity_check:
    print("\nLa transformación es lineal ✅")
else:
    print("\nLa transformación NO es lineal ❌")


# Mostrar resultados
print("Matriz A:\n", A)
print("Determinante de A:", det_A)
print("Traza de A:", trace_A)
print("Eigenvalores de A:", eigenvalues)
print("Eigenvectores de A:\n", eigenvectors)
print("¿La matriz A tiene inversa?:", invertible)

if invertible:
    print("Inversa de A:\n", A_inv)
