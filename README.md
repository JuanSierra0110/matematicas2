# matematicas2
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
# Modificamos la función a x/y
f = x/y

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
dominio_condicion = sp.Ne(y, 0) # y != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = x/y está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Crear una máscara para la región del dominio (donde y != 0)
domain_mask = Y != 0

# Mostrar la región válida (donde y != 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea y = 0)
ax1.axhline(0, color='r', linestyle='--', label='Frontera del Dominio: y = 0')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde y != 0)
# Reemplazamos los valores donde y == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, X/Y, np.nan)

# Para evitar divisiones por cero exactas que puedan causar problemas,
# podemos añadir un pequeño epsilon a Y en el cálculo de Z_surface,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, los puntos donde y=0 serán NaN y no se graficarán.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=10, cstride=10, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = x/y está determinado por la restricción de la división.")
print("Para que la función esté definida, el denominador (y) no puede ser igual a cero.")
print("Esto se expresa como y != 0, lo que gráficamente corresponde a todo el plano (x, y) excepto la línea recta y = 0 (el eje x).")
print("La gráfica 2D muestra esta región sombreada, excluyendo la línea y = 0.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(y + 4*x**2 - 4)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = y + 4*x**2 - 4 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(y + 4*x^2 - 4) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = Y + 4*X**2 - 4

# Crear una máscara para la región del dominio
domain_mask = Z_domain >= 0

ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (una parábola)
# y + 4x^2 - 4 = 0  => y = 4 - 4x^2
x_parabola = np.linspace(-3, 3, 100)
y_parabola = 4 - 4 * x_parabola**2
ax1.plot(x_parabola, y_parabola, 'r--', label='Frontera del Dominio: y = 4 - 4x^2')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 5]) # Ajustar el límite superior para mostrar mejor la parábola


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Crear una malla más fina para la superficie 3D si es necesario, o usar la misma
# x_vals_3d = np.linspace(-3, 3, 100)
# y_vals_3d = np.linspace(-3, 3, 100)
# X_3d, Y_3d = np.meshgrid(x_vals_3d, y_vals_3d)
# Z_domain_3d = Y_3d + 4*X_3d**2 - 4
# domain_mask_3d = Z_domain_3d >= 0
# Z_surface_3d = np.where(domain_mask_3d, np.sqrt(Z_domain_3d), np.nan)


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=10, cstride=10, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(y + 4x^2 - 4) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (y + 4x^2 - 4) debe ser mayor o igual a cero.")
print("Esto se expresa como y + 4x^2 - 4 >= 0, lo que gráficamente corresponde a la región en el plano (x, y) que está por encima o sobre la parábola y = 4 - 4x^2.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = x/y

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
dominio_condicion = sp.Ne(y, 0) # y != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = x/y está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Crear una máscara para la región del dominio (donde y != 0)
domain_mask = Y != 0

# Mostrar la región válida (donde y != 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea y = 0)
ax1.axhline(0, color='r', linestyle='--', label='Frontera del Dominio: y = 0')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])

# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde y != 0)
# Reemplazamos los valores donde y == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, X/Y, np.nan)

# Para evitar divisiones por cero exactas que puedan causar problemas,
# podemos añadir un pequeño epsilon a Y en el cálculo de Z_surface,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, los puntos donde y=0 serán NaN y no se graficarán.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=10, cstride=10, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = x/y está determinado por la restricción de la división.")
print("Para que la función esté definida, el denominador (y) no puede ser igual a cero.")
print("Esto se expresa como y != 0, lo que gráficamente corresponde a todo el plano (x, y) excepto la línea recta y = 0 (el eje x).")
print("La gráfica 2D muestra esta región sombreada, excluyendo la línea y = 0.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.log(x*y - 1)

# 2. Calcula el dominio
# La restricción para un logaritmo es que el argumento sea positivo.
dominio_condicion = x*y - 1 > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = ln(xy - 1) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X*Y - 1

# Crear una máscara para la región del dominio (donde xy - 1 > 0)
domain_mask = Z_domain > 0

# Mostrar la región válida (donde xy - 1 > 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la hipérbola xy = 1)
# Para graficar la hipérbola, necesitamos manejar los casos x > 0 y x < 0 por separado para evitar divisiones por cero
x_hyperbola_pos = np.linspace(0.1, 3, 100) # Evitar x=0
y_hyperbola_pos = 1 / x_hyperbola_pos
ax1.plot(x_hyperbola_pos, y_hyperbola_pos, 'r--', label='Frontera del Dominio: xy = 1')

x_hyperbola_neg = np.linspace(-3, -0.1, 100) # Evitar x=0
y_hyperbola_neg = 1 / x_hyperbola_neg
ax1.plot(x_hyperbola_neg, y_hyperbola_neg, 'r--') # No repetir label


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde xy - 1 > 0)
# Reemplazamos los valores donde xy - 1 <= 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.log(Z_domain), np.nan)


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=10, cstride=10, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = ln(xy - 1) está determinado por la restricción del logaritmo natural.")
print("Para que el logaritmo esté definido, su argumento (xy - 1) debe ser estrictamente mayor que cero.")
print("Esto se expresa como xy - 1 > 0, lo que gráficamente corresponde a las regiones en el plano (x, y) donde el producto x*y es mayor que 1.")
print("Estas regiones están fuera de la hipérbola xy = 1. La gráfica 2D muestra estas regiones sombreadas.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(9 - x**2 - y**2)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = 9 - x**2 - y**2 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(9 - x^2 - y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-4, 4, 400)
y_vals = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = 9 - X**2 - Y**2

# Crear una máscara para la región del dominio (donde 9 - x^2 - y^2 >= 0)
domain_mask = Z_domain >= 0

# Mostrar la región válida (donde 9 - x^2 - y^2 >= 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (un círculo)
# 9 - x^2 - y^2 = 0  => x^2 + y^2 = 9
theta = np.linspace(0, 2*np.pi, 100)
r = 3 # sqrt(9)
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
ax1.plot(x_circle, y_circle, 'r--', label='Frontera del Dominio: x^2 + y^2 = 9')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde 9 - x^2 - y^2 >= 0)
# Reemplazamos los valores donde 9 - x^2 - y^2 < 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos,
# podemos añadir un pequeño epsilon a Z_domain antes de np.sqrt,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, los puntos fuera del dominio serán NaN y no se graficarán.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=10, cstride=10, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(9 - x^2 - y^2) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (9 - x^2 - y^2) debe ser mayor o igual a cero.")
print("Esto se expresa como 9 - x^2 - y^2 >= 0, lo que se puede reescribir como x^2 + y^2 <= 9.")
print("Gráficamente, esto corresponde a todos los puntos (x, y) en el plano cuya distancia al origen (0,0) es menor o igual a 3 (la raíz cuadrada de 9).")
print("Esta región es un círculo cerrado centrado en el origen con un radio de 3.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función (la mitad superior de una esfera de radio 3) solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = (x*y) / (x**2 + y**2)

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
dominio_condicion = sp.Ne(x**2 + y**2, 0) # x^2 + y^2 != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = xy / (x^2 + y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain_denom = X**2 + Y**2

# Crear una máscara para la región del dominio (donde x^2 + y^2 != 0)
# El único punto donde x^2 + y^2 = 0 es el origen (0,0)
domain_mask = Z_domain_denom != 0

# Mostrar la región válida (todo el plano excepto el origen)
# Podemos mostrar todo el plano y luego graficar el punto excluido si es visible a esa escala
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar el punto excluido (el origen)
ax1.plot(0, 0, 'ro', markersize=5, label='Punto excluido del Dominio: (0,0)')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x^2 + y^2 != 0)
# Reemplazamos los valores donde x^2 + y^2 == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, (X*Y) / Z_domain_denom, np.nan)

# Para evitar divisiones por cero exactas que puedan causar problemas,
# podemos añadir un pequeño epsilon al denominador si es necesario,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, el punto en el origen será NaN y no se graficará.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = xy / (x^2 + y^2) está determinado por la restricción de la división.")
print("Para que la función esté definida, el denominador (x^2 + y^2) no puede ser igual a cero.")
print("La expresión x^2 + y^2 es igual a cero si y solo si tanto x como y son cero simultáneamente.")
print("Por lo tanto, el dominio está definido por la condición x^2 + y^2 != 0, lo que gráficamente corresponde a todo el plano (x, y) excepto el origen (el punto (0,0)).")
print("La gráfica 2D muestra la región sombreada, excluyendo el origen.")
print("La gráfica 3D muestra la superficie de la función para todos los puntos excepto el origen.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(1/(x-y))

# 2. Calcula el dominio
# Restricciones:
# 1. El denominador no puede ser cero: x - y != 0 => x != y
# 2. El argumento de la raíz cuadrada debe ser no negativo: 1/(x-y) >= 0
# Combinando ambas, como el numerador es 1 (positivo), el denominador debe ser positivo: x - y > 0 => x > y
dominio_condicion = x - y > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(1/(x-y)) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X - Y

# Crear una máscara para la región del dominio (donde x - y > 0)
domain_mask = Z_domain > 0

# Mostrar la región válida (donde x - y > 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea x = y)
x_line = np.linspace(-3, 3, 100)
y_line = x_line
ax1.plot(x_line, y_line, 'r--', label='Frontera del Dominio: x = y')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x - y > 0)
# Reemplazamos los valores donde x - y <= 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(1/(X-Y)), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos o divisiones por cero,
# np.where y np.nan manejan esto adecuadamente.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(1/(x-y)) está determinado por dos restricciones:")
print("1. El denominador (x - y) no puede ser cero, por lo tanto x != y.")
print("2. El argumento de la raíz cuadrada (1/(x-y)) debe ser mayor o igual a cero. Como el numerador es 1 (positivo), esto implica que el denominador (x - y) debe ser estrictamente mayor que cero, es decir, x - y > 0, lo que se puede reescribir como x > y.")
print("Combinando ambas restricciones, el dominio está definido por la condición x > y.")
print("Gráficamente, esto corresponde a la región en el plano (x, y) que está a la derecha de la línea recta x = y (la línea diagonal que pasa por el origen con pendiente 1).")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.asin(x**2 + y**2 - 1)

# 2. Calcula el dominio
# La restricción para arcsin(u) es que -1 <= u <= 1.
# En este caso, u = x^2 + y^2 - 1.
# Por lo tanto, -1 <= x^2 + y^2 - 1 <= 1.
# Sumando 1 a todas las partes de la desigualdad:
# -1 + 1 <= x^2 + y^2 - 1 + 1 <= 1 + 1
# 0 <= x^2 + y^2 <= 2

dominio_condicion = sp.And(sp.Le(0, x**2 + y**2), sp.Le(x**2 + y**2, 2))
# Simplified condition for plotting: x^2 + y^2 <= 2
dominio_condicion_plot = x**2 + y**2 <= 2


# Expresa la condición del dominio
print("El dominio de la función f(x, y) = arcsin(x^2 + y^2 - 1) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto se simplifica a:")
print(sp.pretty(dominio_condicion_plot))


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X**2 + Y**2

# Crear una máscara para la región del dominio (donde 0 <= x^2 + y^2 <= 2)
domain_mask = np.logical_and(Z_domain >= 0, Z_domain <= 2)

# Mostrar la región válida (donde 0 <= x^2 + y^2 <= 2)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar las fronteras del dominio (círculos)
# x^2 + y^2 = 0 (el origen) - ya está incluido en el sombreado, pero podríamos marcarlo si quisiéramos destacarlo
# x^2 + y^2 = 2
theta = np.linspace(0, 2*np.pi, 100)
r_outer = np.sqrt(2) # sqrt(2)
x_circle_outer = r_outer * np.cos(theta)
y_circle_outer = r_outer * np.sin(theta)
ax1.plot(x_circle_outer, y_circle_outer, 'r--', label='Frontera del Dominio: x^2 + y^2 = 2')

# Marcar el origen como parte de la frontera interior (x^2 + y^2 = 0)
ax1.plot(0, 0, 'ro', markersize=5) # Marcar el origen


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde 0 <= x^2 + y^2 <= 2)
# Reemplazamos los valores fuera del dominio con NaN para no graficarlos
Z_arg = X**2 + Y**2 - 1 # Argumento del arcsin
Z_surface = np.where(domain_mask, np.arcsin(Z_arg), np.nan)


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = arcsin(x^2 + y^2 - 1) está determinado por la restricción del arcsin.")
print("Para que la función arcsin(u) esté definida, el argumento 'u' debe estar en el intervalo cerrado [-1, 1].")
print("En este caso, el argumento es u = x^2 + y^2 - 1.")
print("Por lo tanto, la restricción es -1 <= x^2 + y^2 - 1 <= 1.")
print("Al sumar 1 a todos los lados de la desigualdad, obtenemos 0 <= x^2 + y^2 <= 2.")
print("La condición x^2 + y^2 >= 0 siempre es cierta para números reales x e y.")
print("La condición x^2 + y^2 <= 2 gráficamente corresponde a todos los puntos (x, y) en el plano cuya distancia al origen (0,0) es menor o igual a sqrt(2).")
print("Esta región es un círculo cerrado centrado en el origen con un radio de sqrt(2).")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.log(4 - x**2 + y)

# 2. Calcula el dominio
# La restricción para un logaritmo es que el argumento sea positivo.
dominio_condicion = 4 - x**2 + y > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = ln(4 - x^2 + y) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = 4 - X**2 + Y

# Crear una máscara para la región del dominio (donde 4 - x^2 + y > 0)
domain_mask = Z_domain > 0

# Mostrar la región válida (donde 4 - x^2 + y > 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (una parábola)
# 4 - x^2 + y = 0  => y = x^2 - 4
x_parabola = np.linspace(-3, 3, 100)
y_parabola = x_parabola**2 - 4
ax1.plot(x_parabola, y_parabola, 'r--', label='Frontera del Dominio: y = x^2 - 4')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-5, 3]) # Ajustar el límite inferior para mostrar mejor la parábola


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde 4 - x^2 + y > 0)
# Reemplazamos los valores donde 4 - x^2 + y <= 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.log(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.log de números no positivos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = ln(4 - x^2 + y) está determinado por la restricción del logaritmo natural.")
print("Para que el logaritmo esté definido, su argumento (4 - x^2 + y) debe ser estrictamente mayor que cero.")
print("Esto se expresa como 4 - x^2 + y > 0, lo que se puede reescribir como y > x^2 - 4.")
print("Gráficamente, esto corresponde a la región en el plano (x, y) que está por encima de la parábola y = x^2 - 4.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(x - 2*y)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = x - 2*y >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(x - 2y) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X - 2*Y

# Crear una máscara para la región del dominio (donde x - 2y >= 0)
domain_mask = Z_domain >= 0

# Mostrar la región válida (donde x - 2y >= 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (una línea recta)
# x - 2y = 0  => x = 2y => y = x/2
x_line = np.linspace(-3, 3, 100)
y_line = x_line / 2
ax1.plot(x_line, y_line, 'r--', label='Frontera del Dominio: x - 2y = 0')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x - 2y >= 0)
# Reemplazamos los valores donde x - 2y < 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(x - 2y) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (x - 2y) debe ser mayor o igual a cero.")
print("Esto se expresa como x - 2y >= 0, lo que se puede reescribir como x >= 2y o y <= x/2.")
print("Gráficamente, esto corresponde a la región en el plano (x, y) que está por debajo o sobre la línea recta y = x/2.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(x*y - 5)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = x*y - 5 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(xy - 5) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X*Y - 5

# Crear una máscara para la región del dominio (donde xy - 5 >= 0)
domain_mask = Z_domain >= 0

# Mostrar la región válida (donde xy - 5 >= 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la hipérbola xy = 5)
# Para graficar la hipérbola, necesitamos manejar los casos x > 0 y x < 0 por separado para evitar divisiones por cero
x_hyperbola_pos = np.linspace(1, 5, 100) # Evitar x=0 o valores cercanos
y_hyperbola_pos = 5 / x_hyperbola_pos
ax1.plot(x_hyperbola_pos, y_hyperbola_pos, 'r--', label='Frontera del Dominio: xy = 5')

x_hyperbola_neg = np.linspace(-5, -1, 100) # Evitar x=0 o valores cercanos
y_hyperbola_neg = 5 / x_hyperbola_neg
ax1.plot(x_hyperbola_neg, y_hyperbola_neg, 'r--') # No repetir label


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde xy - 5 >= 0)
# Reemplazamos los valores donde xy - 5 < 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(xy - 5) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (xy - 5) debe ser mayor o igual a cero.")
print("Esto se expresa como xy - 5 >= 0, lo que se puede reescribir como xy >= 5.")
print("Gráficamente, esto corresponde a las regiones en el plano (x, y) donde el producto x*y es mayor o igual a 5.")
print("Estas regiones están fuera de la hipérbola xy = 5, incluyendo la hipérbola misma. La gráfica 2D muestra estas regiones sombreadas.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y, z = sp.symbols('x y z')
f = sp.sqrt(z - x**2 - y**2)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = z - x**2 - y**2 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y, z) = sqrt(z - x^2 - y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto se puede reescribir como: z >= x^2 + y^2")

# 3. Genera el gráfico 3D del dominio
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear una malla de puntos en 3D para visualizar el dominio
# Limitamos los rangos para una visualización razonable
x_vals = np.linspace(-3, 3, 50)
y_vals = np.linspace(-3, 3, 50)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
z_surface = x_mesh**2 + y_mesh**2 # La frontera del dominio es z = x^2 + y^2

# Para mostrar la región z >= x^2 + y^2, podemos graficar la superficie z = x^2 + y^2
# y conceptualizar la región por encima de ella.
# Alternativamente, podemos mostrar puntos aleatorios dentro del dominio o usar un volumen renderizado (más complejo).
# Graficaremos la superficie frontera para delimitar visualmente la región.

ax.plot_surface(x_mesh, y_mesh, z_surface, cmap='viridis', alpha=0.6, rstride=5, cstride=5)

# Añadir puntos de ejemplo dentro del dominio para ilustrar la región
# Elegimos puntos donde z es mayor que x^2 + y^2
num_points = 1000
random_x = np.random.uniform(-3, 3, num_points)
random_y = np.random.uniform(-3, 3, num_points)
random_z_min = random_x**2 + random_y**2
random_z = np.random.uniform(random_z_min, random_z_min + 5, num_points) # Z por encima de la superficie

ax.scatter(random_x, random_y, random_z, c='grey', marker='.', alpha=0.2)


ax.set_title('Dominio de f(x, y, z) = sqrt(z - x^2 - y^2) en R³')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 10]) # Ajustar límite Z para mostrar mejor la región

plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y, z) = sqrt(z - x^2 - y^2) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (z - x^2 - y^2) debe ser mayor o igual a cero.")
print("Esto se expresa como z - x^2 - y^2 >= 0, lo que se puede reescribir como z >= x^2 + y^2.")
print("Gráficamente, esto corresponde a la región en el espacio 3D (x, y, z) que está por encima o sobre el paraboloide elíptico z = x^2 + y^2.")
print("La gráfica 3D muestra la superficie de este paraboloide, indicando la frontera inferior del dominio. La región válida es todo el espacio por encima de esta superficie.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = 1 / (x**2 - y**2)

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
dominio_condicion = sp.Ne(x**2 - y**2, 0) # x^2 - y^2 != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = 1 / (x^2 - y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain_denom = X**2 - Y**2

# Crear una máscara para la región del dominio (donde x^2 - y^2 != 0)
# Esto corresponde a donde X != Y y X != -Y
domain_mask = Z_domain_denom != 0

# Mostrar la región válida (todo el plano excepto las líneas y = x y y = -x)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar las fronteras del dominio (las líneas y = x y y = -x)
x_line = np.linspace(-3, 3, 100)
ax1.plot(x_line, x_line, 'r--', label='Frontera del Dominio: y = x')
ax1.plot(x_line, -x_line, 'r--', label='Frontera del Dominio: y = -x')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x^2 - y^2 != 0)
# Reemplazamos los valores donde x^2 - y^2 == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, 1 / Z_domain_denom, np.nan)

# Para evitar divisiones por cero exactas o valores muy grandes cerca de la frontera,
# podemos añadir un pequeño epsilon al denominador si es necesario,
# o simplemente confiar en que np.where y np.nan manejan esto.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = 1 / (x^2 - y^2) está determinado por la restricción del denominador.")
print("Para que la función esté definida, el denominador (x^2 - y^2) no puede ser igual a cero.")
print("Esto se expresa como x^2 - y^2 != 0, que se puede factorizar como (x - y)(x + y) != 0.")
print("Esto implica que x - y != 0 (es decir, x != y) y x + y != 0 (es decir, x != -y).")
print("Gráficamente, esto corresponde a todo el plano (x, y) excepto las líneas rectas y = x y y = -x.")
print("La gráfica 2D muestra la región sombreada, excluyendo estas dos líneas.")
print("La gráfica 3D muestra la superficie de la función para todos los puntos excepto los que se encuentran en estas líneas.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(y - x**2)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = y - x**2 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(y - x^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 10, 400) # Adjusted y range to better show the parabola
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = Y - X**2

# Crear una máscara para la región del dominio (donde y - x^2 >= 0)
domain_mask = Z_domain >= 0

# Mostrar la región válida (donde y - x^2 >= 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (una parábola)
# y - x^2 = 0  => y = x^2
x_parabola = np.linspace(-3, 3, 100)
y_parabola = x_parabola**2
ax1.plot(x_parabola, y_parabola, 'r--', label='Frontera del Dominio: y = x^2')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-1, 10]) # Adjusted y limits for better visualization


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde y - x^2 >= 0)
# Reemplazamos los valores donde y - x^2 < 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(y - x^2) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (y - x^2) debe ser mayor o igual a cero.")
print("Esto se expresa como y - x^2 >= 0, lo que se puede reescribir como y >= x^2.")
print("Gráficamente, esto corresponde a la región en el plano (x, y) que está por encima o sobre la parábola y = x^2.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.log(x + y)

# 2. Calcula el dominio
# La restricción para un logaritmo es que el argumento sea positivo.
dominio_condicion = x + y > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = ln(x + y) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = X + Y

# Crear una máscara para la región del dominio (donde x + y > 0)
domain_mask = Z_domain > 0

# Mostrar la región válida (donde x + y > 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea x + y = 0 => y = -x)
x_line = np.linspace(-5, 5, 100)
y_line = -x_line
ax1.plot(x_line, y_line, 'r--', label='Frontera del Dominio: x + y = 0')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x + y > 0)
# Reemplazamos los valores donde x + y <= 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.log(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.log de números no positivos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = ln(x + y) está determinado por la restricción del logaritmo natural.")
print("Para que el logaritmo esté definido, su argumento (x + y) debe ser estrictamente mayor que cero.")
print("Esto se expresa como x + y > 0, lo que se puede reescribir como y > -x.")
print("Gráficamente, esto corresponde a la región en el plano (x, y) que está por encima de la línea recta y = -x.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.atan(y/x)

# 2. Calcula el dominio
# La restricción para la división es que el denominador no sea cero.
dominio_condicion = sp.Ne(x, 0) # x != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = arctan(y/x) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Crear una máscara para la región del dominio (donde x != 0)
domain_mask = X != 0

# Mostrar la región válida (donde x != 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea x = 0)
ax1.axvline(0, color='r', linestyle='--', label='Frontera del Dominio: x = 0')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x != 0)
# Reemplazamos los valores donde x == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.arctan(Y/X), np.nan)

# Para evitar divisiones por cero exactas que puedan causar problemas,
# podemos añadir un pequeño epsilon a X en el cálculo de Z_surface,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, los puntos donde x=0 serán NaN y no se graficarán.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = arctan(y/x) está determinado por la restricción del denominador dentro del argumento del arctan.")
print("Para que la división y/x esté definida, el denominador (x) no puede ser igual a cero.")
print("Esto se expresa como x != 0, lo que gráficamente corresponde a todo el plano (x, y) excepto la línea recta x = 0 (el eje y).")
print("La gráfica 2D muestra esta región sombreada, excluyendo la línea x = 0.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(x*y / (x + y + 1))

# 2. Calcula el dominio
# Restricciones:
# 1. El denominador no puede ser cero: x + y + 1 != 0 => y != -x - 1
# 2. El argumento de la raíz cuadrada debe ser no negativo: xy / (x + y + 1) >= 0
# Esto ocurre cuando (xy >= 0 AND x + y + 1 > 0) OR (xy <= 0 AND x + y + 1 < 0)
# Caso 1: (x >= 0 AND y >= 0 AND y > -x - 1) OR (x <= 0 AND y <= 0 AND y > -x - 1)
# Caso 2: (x >= 0 AND y <= 0 AND y < -x - 1) OR (x <= 0 AND y >= 0 AND y < -x - 1)

dominio_condicion_denom = sp.Ne(x + y + 1, 0)
dominio_condicion_sqrt = sp.Ge(x*y / (x + y + 1), 0)

dominio_condicion = sp.And(dominio_condicion_denom, dominio_condicion_sqrt)

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(xy / (x + y + 1)) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluar las condiciones del dominio en la malla
denom_vals = X + Y + 1
sqrt_arg_vals = (X * Y) / denom_vals

# Crear una máscara para la región del dominio
# Evitar divisiones por cero al evaluar la condición de la raíz cuadrada
# Solo evaluamos sqrt_arg_vals donde denom_vals != 0
domain_mask = np.zeros_like(X, dtype=bool)
non_zero_denom_mask = denom_vals != 0

# Evaluar la condición de la raíz cuadrada solo donde el denominador no es cero
domain_mask[non_zero_denom_mask] = sqrt_arg_vals[non_zero_denom_mask] >= 0

# Asegurarse de que los puntos donde el denominador es cero estén fuera del dominio
domain_mask[~non_zero_denom_mask] = False


# Mostrar la región válida
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar las fronteras del dominio
# Frontera 1: denominador = 0 => y = -x - 1
x_line_denom = np.linspace(-5, 5, 100)
y_line_denom = -x_line_denom - 1
ax1.plot(x_line_denom, y_line_denom, 'r--', label='Frontera: x + y + 1 = 0')

# Fronteras 2 y 3: xy = 0 => x = 0 o y = 0
ax1.axvline(0, color='b', linestyle='--', label='Frontera: x = 0')
ax1.axhline(0, color='g', linestyle='--', label='Frontera: y = 0')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio
# Reemplazamos los valores fuera del dominio con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(sqrt_arg_vals), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos o divisiones por cero,
# np.where y np.nan manejan esto adecuadamente.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(xy / (x + y + 1)) está determinado por dos restricciones principales:")
print("1. El denominador (x + y + 1) no puede ser cero, es decir, x + y + 1 != 0 (o y != -x - 1). Esto excluye la línea recta y = -x - 1.")
print("2. El argumento de la raíz cuadrada (xy / (x + y + 1)) debe ser mayor o igual a cero. Esto ocurre cuando el numerador y el denominador tienen el mismo signo (ambos positivos o ambos negativos), o cuando el numerador es cero (xy = 0).")
print("   - Caso 1: xy >= 0 y x + y + 1 > 0. Esto corresponde a los cuadrantes I y III (donde xy >= 0) *por encima* de la línea y = -x - 1.")
print("   - Caso 2: xy <= 0 y x + y + 1 < 0. Esto corresponde a los cuadrantes II y IV (donde xy <= 0) *por debajo* de la línea y = -x - 1.")
print("La gráfica 2D muestra la región sombreada que cumple estas condiciones. Las fronteras del dominio son las líneas x = 0, y = 0 y y = -x - 1.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(1 - (x - 2)**2 - (y + 1)**2)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
dominio_condicion = 1 - (x - 2)**2 - (y + 1)**2 >= 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(1 - (x - 2)^2 - (y + 1)^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto se puede reescribir como: (x - 2)^2 + (y + 1)^2 <= 1")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(0, 4, 400) # Ajustar rango x para centrar en 2
y_vals = np.linspace(-3, 1, 400) # Ajustar rango y para centrar en -1
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = 1 - (X - 2)**2 - (Y + 1)**2

# Crear una máscara para la región del dominio (donde 1 - (x - 2)^2 - (y + 1)^2 >= 0)
domain_mask = Z_domain >= 0

# Mostrar la región válida (donde 1 - (x - 2)^2 - (y + 1)^2 >= 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (un círculo)
# 1 - (x - 2)^2 - (y + 1)^2 = 0  => (x - 2)^2 + (y + 1)^2 = 1
theta = np.linspace(0, 2*np.pi, 100)
r = 1 # sqrt(1)
center_x = 2
center_y = -1
x_circle = center_x + r * np.cos(theta)
y_circle = center_y + r * np.sin(theta)
ax1.plot(x_circle, y_circle, 'r--', label='Frontera del Dominio: (x - 2)^2 + (y + 1)^2 = 1')

ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([0, 4])
ax1.set_ylim([-3, 1])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde 1 - (x - 2)^2 - (y + 1)^2 >= 0)
# Reemplazamos los valores donde 1 - (x - 2)^2 - (y + 1)^2 < 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos,
# np.where y np.nan manejan esto adecuadamente.

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(1 - (x - 2)^2 - (y + 1)^2) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (1 - (x - 2)^2 - (y + 1)^2) debe ser mayor o igual a cero.")
print("Esto se expresa como 1 - (x - 2)^2 - (y + 1)^2 >= 0, lo que se puede reescribir como (x - 2)^2 + (y + 1)^2 <= 1.")
print("Gráficamente, esto corresponde a todos los puntos (x, y) en el plano cuya distancia al punto (2, -1) es menor o igual a 1 (la raíz cuadrada de 1).")
print("Esta región es un círculo cerrado centrado en (2, -1) con un radio de 1.")
print("La gráfica 2D muestra esta región sombreada.")
print("La gráfica 3D muestra la superficie de la función (la mitad superior de una esfera de radio 1 centrada en (2, -1, 0)) solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y, z = sp.symbols('x y z')
f = sp.log(x*y*z)

# 2. Calcula el dominio
# La restricción para un logaritmo es que el argumento sea positivo.
dominio_condicion = x*y*z > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y, z) = ln(xyz) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# Explicación del dominio en R³
print("\nExplicación del Dominio en R³:")
print("El dominio de la función f(x, y, z) = ln(xyz) está determinado por la restricción del logaritmo natural.")
print("Para que el logaritmo esté definido, su argumento (x*y*z) debe ser estrictamente mayor que cero.")
print("Esto ocurre cuando el producto de las tres variables es positivo.")
print("Hay cuatro casos posibles para que esto suceda:")
print("1. x > 0, y > 0, z > 0 (El primer octante)")
print("2. x > 0, y < 0, z < 0")
print("3. x < 0, y > 0, z < 0")
print("4. x < 0, y < 0, z > 0")
print("Gráficamente, el dominio corresponde a la unión de estos cuatro octantes del espacio tridimensional.")

# 3. Genera el gráfico 3D del dominio (visualizando la región)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Creamos puntos de ejemplo en cada uno de los octantes del dominio
num_points = 5000
# Octante 1 (+++)
ax.scatter(np.random.uniform(0.1, 3, num_points), np.random.uniform(0.1, 3, num_points), np.random.uniform(0.1, 3, num_points), c='skyblue', marker='.', alpha=0.3, label='Octante 1 (x>0, y>0, z>0)')
# Octante 2 (+--)
ax.scatter(np.random.uniform(0.1, 3, num_points), np.random.uniform(-3, -0.1, num_points), np.random.uniform(-3, -0.1, num_points), c='lightcoral', marker='.', alpha=0.3, label='Octante 2 (x>0, y<0, z<0)')
# Octante 3 (-+-)
ax.scatter(np.random.uniform(-3, -0.1, num_points), np.random.uniform(0.1, 3, num_points), np.random.uniform(-3, -0.1, num_points), c='lightgreen', marker='.', alpha=0.3, label='Octante 3 (x<0, y>0, z<0)')
# Octante 4 (--+)
ax.scatter(np.random.uniform(-3, -0.1, num_points), np.random.uniform(-3, -0.1, num_points), np.random.uniform(0.1, 3, num_points), c='orchid', marker='.', alpha=0.3, label='Octante 4 (x<0, y<0, z>0)')


ax.set_title('Dominio de f(x, y, z) = ln(xyz) en R³')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.legend()

plt.show()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(1 / (4 - x**2 - y**2))

# 2. Calcula el dominio
# Restricciones:
# 1. El denominador no puede ser cero: 4 - x**2 - y**2 != 0
# 2. El argumento de la raíz cuadrada debe ser no negativo: 1 / (4 - x**2 - y**2) >= 0
# Combinando ambas, el denominador debe ser estrictamente positivo: 4 - x**2 - y**2 > 0
dominio_condicion = 4 - x**2 - y**2 > 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(1 / (4 - x^2 - y^2)) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto se puede reescribir como: x^2 + y^2 < 4")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z_domain = 4 - X**2 - Y**2

# Crear una máscara para la región del dominio (donde 4 - x^2 - y^2 > 0)
domain_mask = Z_domain > 0

# Mostrar la región válida (donde 4 - x^2 - y^2 > 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (un círculo)
# 4 - x^2 - y^2 = 0  => x^2 + y^2 = 4
theta = np.linspace(0, 2*np.pi, 100)
r = 2 # sqrt(4)
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
ax1.plot(x_circle, y_circle, 'r--', label='Frontera del Dominio: x^2 + y^2 = 4')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde 4 - x^2 - y^2 > 0)
# Reemplazamos los valores donde 4 - x^2 - y**2 <= 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.sqrt(1 / Z_domain), np.nan)

# Para evitar RuntimeWarning con np.sqrt de números negativos o divisiones por cero,
# np.where y np.nan manejan esto adecuadamente.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(1 / (4 - x^2 - y^2)) está determinado por dos restricciones principales:")
print("1. El denominador (4 - x^2 - y^2) no puede ser cero.")
print("2. El argumento de la raíz cuadrada (1 / (4 - x^2 - y^2)) debe ser mayor o igual a cero.")
print("Combinando ambas, el denominador debe ser estrictamente positivo: 4 - x^2 - y^2 > 0.")
print("Esto se puede reescribir como x^2 + y^2 < 4.")
print("Gráficamente, esto corresponde a todos los puntos (x, y) en el plano cuya distancia al origen (0,0) es estrictamente menor que 2 (la raíz cuadrada de 4).")
print("Esta región es un círculo abierto (sin incluir la frontera) centrado en el origen con un radio de 2.")
print("La gráfica 2D muestra esta región sombreada, excluyendo la frontera circular.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sin(x*y)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función sin(xy).
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sin(xy) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.sin(X*Y)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = sin(xy)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sin(xy) está determinado por las restricciones de sus componentes.")
print("La función seno está definida para todos los números reales, y el producto x*y también está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = x**2 + y**2

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función x^2 + y^2.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = x^2 + y^2 está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = X**2 + Y**2

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = x^2 + y^2')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = x^2 + y^2 está determinado por las restricciones de sus componentes.")
print("La suma de cuadrados x^2 + y^2 está definida para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (un paraboloide elíptico) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = 9 - x**2 + y**2

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función 9 - x^2 + y^2.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = 9 - X**2 + Y**2

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = 9 - x^2 + y^2')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está determinado por las restricciones de sus componentes.")
print("La resta y suma de cuadrados y constantes (9 - x^2 + y^2) está definida para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (un paraboloide hiperbólico invertido) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = 9 - x**2 + y**2

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función 9 - x^2 + y^2.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = 9 - X**2 + Y**2

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = 9 - x^2 + y^2')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está determinado por las restricciones de sus componentes.")
print("La resta y suma de cuadrados y constantes (9 - x^2 + y^2) está definida para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (un paraboloide hiperbólico invertido) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = 9 - x**2 + y**2

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función 9 - x^2 + y^2.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = 9 - X**2 + Y**2

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = 9 - x^2 + y^2')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = 9 - x^2 + y^2 está determinado por las restricciones de sus componentes.")
print("La resta y suma de cuadrados y constantes (9 - x^2 + y^2) está definida para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (un paraboloide hiperbólico invertido) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(x**2 + y**2)

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
# x^2 + y^2 >= 0
# Esto siempre es cierto para números reales x e y.
dominio_condicion = sp.Ge(x**2 + y**2, 0) # x^2 + y^2 >= 0
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(x^2 + y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.sqrt(X**2 + Y**2)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = sqrt(x^2 + y^2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(x^2 + y^2) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (x^2 + y^2) debe ser mayor o igual a cero.")
print("La expresión x^2 + y^2 siempre es mayor o igual a cero para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (un cono elíptico) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.exp(-x**2 - y**2)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función exponencial.
# El argumento -x^2 - y^2 está definido para todos los números reales x e y.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = e^(-x^2 - y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.exp(-X**2 - Y**2)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = e^(-x^2 - y^2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = e^(-x^2 - y^2) está determinado por las restricciones de sus componentes.")
print("La función exponencial está definida para todos los números reales, y el argumento (-x^2 - y^2) también está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (una campana gaussiana en 3D) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sin(x) * sp.cos(y)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en las funciones seno y coseno.
# El producto de sin(x) y cos(y) está definido para todos los números reales x e y.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sin(x)cos(y) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-np.pi, np.pi, 400)
y_vals = np.linspace(-np.pi, np.pi, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-np.pi, np.pi])
ax1.set_ylim([-np.pi, np.pi])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.sin(X) * np.cos(Y)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = sin(x)cos(y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sin(x)cos(y) está determinado por las restricciones de sus componentes.")
print("Las funciones seno y coseno están definidas para todos los números reales, y su producto también está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = 1 / (1 + x**2 + y**2)

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
# El denominador 1 + x^2 + y^2 siempre es >= 1 para números reales x e y.
# Por lo tanto, el denominador nunca es cero, y no hay restricciones.
dominio_condicion = sp.Ne(1 + x**2 + y**2, 0)
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = 1 / (1 + x^2 + y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = 1 / (1 + X**2 + Y**2)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = 1 / (1 + x^2 + y^2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = 1 / (1 + x^2 + y^2) está determinado por la restricción del denominador.")
print("Para que la función esté definida, el denominador (1 + x^2 + y^2) no puede ser igual a cero.")
print("La expresión 1 + x^2 + y^2 siempre es mayor o igual a 1 para todos los números reales x e y, ya que x^2 y y^2 son no negativos.")
print("Por lo tanto, el denominador nunca es cero y no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (una campana simétrica) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.log(x**2 + y**2 + 1)

# 2. Calcula el dominio
# La restricción para un logaritmo es que el argumento sea positivo.
# x^2 + y^2 + 1 > 0
# Como x^2 >= 0 and y^2 >= 0, x^2 + y^2 >= 0.
# Adding 1, x^2 + y^2 + 1 >= 1.
# Thus, x^2 + y^2 + 1 is always positive.
dominio_condicion = sp.Gt(x**2 + y**2 + 1, 0)
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = ln(x^2 + y^2 + 1) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.log(X**2 + Y**2 + 1)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = ln(x^2 + y^2 + 1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = ln(x^2 + y^2 + 1) está determinado por la restricción del logaritmo natural.")
print("Para que el logaritmo esté definido, su argumento (x^2 + y^2 + 1) debe ser estrictamente mayor que cero.")
print("La expresión x^2 + y^2 siempre es mayor o igual a cero para todos los números reales x e y. Por lo tanto, x^2 + y^2 + 1 siempre es mayor o igual a 1, y así siempre es positivo.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = x * sp.exp(-y**2)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función x * e^(-y^2).
# x está definido para todos los números reales.
# e^(-y^2) está definido para todos los números reales y.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = xe^(-y^2) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = X * np.exp(-Y**2)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = xe^(-y^2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = xe^(-y^2) está determinado por las restricciones de sus componentes.")
print("La variable x está definida para todos los números reales.")
print("La función exponencial e^(-y^2) está definida para todos los números reales y.")
print("El producto de estas dos funciones está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = (x**2 - y**2) / (x**2 + y**2 + 1)

# 2. Calcula el dominio
# La restricción para una división es que el denominador no sea cero.
# El denominador x^2 + y^2 + 1 siempre es >= 1 para números reales x e y.
# Por lo tanto, el denominador nunca es cero, y no hay restricciones.
dominio_condicion = sp.Ne(x**2 + y**2 + 1, 0)
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = (x^2 - y^2) / (x^2 + y^2 + 1) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = (X**2 - Y**2) / (X**2 + Y**2 + 1)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = (x^2 - y^2) / (x^2 + y^2 + 1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = (x^2 - y^2) / (x^2 + y^2 + 1) está determinado por la restricción del denominador.")
print("Para que la función esté definida, el denominador (x^2 + y^2 + 1) no puede ser igual a cero.")
print("La expresión x^2 + y^2 + 1 siempre es mayor o igual a 1 para todos los números reales x e y, ya que x^2 y y^2 son no negativos.")
print("Por lo tanto, el denominador nunca es cero y no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = x**2*y + y**3

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función polinómica x^2*y + y^3.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = x^2y + y^3 está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = X**2 * Y + Y**3

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = x^2y + y^3')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = x^2y + y^3 está determinado por las restricciones de sus componentes.")
print("Esta es una función polinómica, y las operaciones de multiplicación y suma de potencias están definidas para todos los números reales x e y.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sin(sp.sqrt(x**2 + y**2))

# 2. Calcula el dominio
# La restricción para la raíz cuadrada es que el argumento sea no negativo: x^2 + y^2 >= 0
# Esto siempre es cierto para números reales x e y.
# La función seno está definida para todos los números reales.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.Ge(x**2 + y**2, 0)
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sin(sqrt(x^2 + y^2)) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")


# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-5, 5, 400)
y_vals = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
R = np.sqrt(X**2 + Y**2) # Distancia al origen
Z_surface = np.sin(R)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = sin(sqrt(x^2 + y^2))')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sin(sqrt(x^2 + y^2)) está determinado por las restricciones de sus componentes.")
print("La restricción para la raíz cuadrada es que el argumento (x^2 + y^2) debe ser mayor o igual a cero, lo cual siempre es cierto para números reales x e y.")
print("La función seno está definida para todos los números reales.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función (una onda circular) sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = y * sp.cos(x)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función y * cos(x).
# y está definido para todos los números reales.
# cos(x) está definido para todos los números reales x.
# El producto de estas dos funciones está definido para todos los pares ordenados (x, y) en el plano R^2.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = y*cos(x) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-2*np.pi, 2*np.pi, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-2*np.pi, 2*np.pi])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = Y * np.cos(X)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = y*cos(x)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = y*cos(x) está determinado por las restricciones de sus componentes.")
print("La variable y está definida para todos los números reales.")
print("La función coseno cos(x) está definida para todos los números reales x.")
print("El producto de estas dos funciones está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.exp(-sp.Abs(x + y))

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en la función exponencial o el valor absoluto.
# El argumento -|x+y| está definido para todos los números reales x e y.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = e^(-|x+y|) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.exp(-np.abs(X + Y))

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = e^(-|x+y|)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = e^(-|x+y|) está determinado por las restricciones de sus componentes.")
print("La función exponencial está definida para todos los números reales, y el argumento -|x+y| también está definido para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.atan(y/x)

# 2. Calcula el dominio
# La restricción para la división es que el denominador no sea cero.
dominio_condicion = sp.Ne(x, 0) # x != 0

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = arctan(y/x) está definido por la condición:")
print(sp.pretty(dominio_condicion))

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Crear una máscara para la región del dominio (donde x != 0)
domain_mask = X != 0

# Mostrar la región válida (donde x != 0)
ax1.imshow(domain_mask, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='Greys', alpha=0.3)

# Graficar la frontera del dominio (la línea x = 0)
ax1.axvline(0, color='r', linestyle='--', label='Frontera del Dominio: x = 0')


ax1.set_title('Dominio en el plano (x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])


# b) Gráfica 3D de la superficie z = f(x, y) limitada al dominio
ax2 = fig.add_subplot(122, projection='3d')

# Solo calculamos Z para los puntos dentro del dominio (donde x != 0)
# Reemplazamos los valores donde x == 0 con NaN para no graficarlos
Z_surface = np.where(domain_mask, np.arctan(Y/X), np.nan)

# Para evitar divisiones por cero exactas que puedan causar problemas,
# podemos añadir un pequeño epsilon a X en el cálculo de Z_surface,
# o simplemente confiar en que np.where y np.nan manejan esto.
# Con np.where y np.nan, los puntos donde x=0 serán NaN y no se graficarán.


surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) en el dominio')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = arctan(y/x) está determinado por la restricción del denominador dentro del argumento del arctan.")
print("Para que la división y/x esté definida, el denominador (x) no puede ser igual a cero.")
print("Esto se expresa como x != 0, lo que gráficamente corresponde a todo el plano (x, y) excepto la línea recta x = 0 (el eje y).")
print("La gráfica 2D muestra esta región sombreada, excluyendo la línea x = 0.")
print("La gráfica 3D muestra la superficie de la función solo para los puntos que cumplen esta condición de dominio.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.sqrt(sp.Abs(x*y))

# 2. Calcula el dominio
# La restricción para una raíz cuadrada es que el argumento sea no negativo.
# |xy| >= 0
# El valor absoluto de cualquier número real siempre es mayor o igual a cero.
# Por lo tanto, la condición |xy| >= 0 siempre es cierta para números reales x e y.
dominio_condicion = sp.Ge(sp.Abs(x*y), 0)
# SymPy simplificará esto a True

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = sqrt(|xy|) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.sqrt(np.abs(X*Y))

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = sqrt(|xy|)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = sqrt(|xy|) está determinado por la restricción de la raíz cuadrada.")
print("Para que la función esté definida, el argumento de la raíz cuadrada (|xy|) debe ser mayor o igual a cero.")
print("La expresión |xy| siempre es mayor o igual a cero para todos los números reales x e y, ya que el valor absoluto siempre es no negativo.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define la función simbólicamente
x, y = sp.symbols('x y')
f = sp.cos(x) + sp.sin(y)

# 2. Calcula el dominio
# No hay restricciones (raíces cuadradas, denominadores, logaritmos) en las funciones coseno y seno.
# La suma de cos(x) y sin(y) está definida para todos los números reales x e y.
# Por lo tanto, el dominio es todo R^2.
dominio_condicion = sp.true # Representa la condición siempre verdadera

# Expresa la condición del dominio
print("El dominio de la función f(x, y) = cos(x) + sin(y) está definido por la condición:")
print(sp.pretty(dominio_condicion))
print("\nEsto significa que el dominio es todo el plano R^2.")

# 3. Genera los gráficos
fig = plt.figure(figsize=(12, 6))

# a) Gráfico 2D del dominio (todo el plano R^2)
ax1 = fig.add_subplot(121)
x_vals = np.linspace(-2*np.pi, 2*np.pi, 400)
y_vals = np.linspace(-2*np.pi, 2*np.pi, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Como el dominio es todo R^2, podemos simplemente mostrar el plano completo.
# No hay regiones inválidas que sombrear.

ax1.set_title('Dominio en el plano (x, y) (Todo R²)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([-2*np.pi, 2*np.pi])
ax1.set_ylim([-2*np.pi, 2*np.pi])
# Opcional: añadir un texto o un sombreado ligero para indicar todo el plano es el dominio
ax1.text(0, 0, 'Dominio: Todo R²', horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')


# b) Gráfica 3D de la superficie z = f(x, y) en el dominio (todo R^2)
ax2 = fig.add_subplot(122, projection='3d')

# Calculamos Z para todos los puntos en la malla
Z_surface = np.cos(X) + np.sin(Y)

surf = ax2.plot_surface(X, Y, Z_surface, cmap='viridis', rstride=20, cstride=20, linewidth=0, antialiased=False)

ax2.set_title('Gráfica 3D de z = f(x, y) = cos(x) + sin(y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

# 4. Explica el dominio hallado y las restricciones principales.
print("\nExplicación del Dominio:")
print("El dominio de la función f(x, y) = cos(x) + sin(y) está determinado por las restricciones de sus componentes.")
print("Las funciones coseno y seno están definidas para todos los números reales, y su suma también está definida para todos los pares ordenados (x, y) en el plano R^2.")
print("Por lo tanto, no hay restricciones en los valores de x e y para que la función esté definida.")
print("El dominio es, por lo tanto, todo el plano R^2.")
print("La gráfica 2D ilustra esto mostrando el plano completo como la región del dominio.")
print("La gráfica 3D muestra la superficie de la función sobre todo el plano R^2 dentro del rango de visualización definido.")

