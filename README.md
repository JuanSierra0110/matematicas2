# matematicas2
# Código reconstruido del notebook 'lista_4_matematicas.ipynb'
# He reunido y ordenado las celdas de código que estaban en el notebook proporcionado.
# Si necesitas el .ipynb exacto (con outputs/metadata) pásame el archivo .ipynb completo o indícame cómo quieres que lo entregue.

from sympy import symbols, sqrt, log, sin
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Configuración común
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")


def problema_1_sqrt_xyz():
    """
    f(x,y,z) = sqrt(z - x^2 - y^2)
    Dominio: z >= x^2 + y^2 (región arriba o sobre el paraboloide z = x^2 + y^2)
    Visualización del paraboloide (frontera) y texto indicando la región válida.
    """
    print("Problema 1: f(x, y, z) = sqrt(z - x^2 - y^2)")
    x, y, z = symbols('x y z')
    f = sqrt(z - x**2 - y**2)

    print("La función f(x, y, z) = sqrt(z - x^2 - y^2) está definida cuando z - x^2 - y^2 >= 0.")
    print("Dominio: z >= x^2 + y^2 (región arriba o sobre el paraboloide z = x^2 + y^2).")

    # Visualización (opcional)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Dominio de f(x, y, z) = sqrt(z - x^2 - y^2)')

    x_boundary = np.linspace(-5, 5, 50)
    y_boundary = np.linspace(-5, 5, 50)
    X_boundary, Y_boundary = np.meshgrid(x_boundary, y_boundary)
    Z_boundary = X_boundary**2 + Y_boundary**2

    ax.plot_surface(X_boundary, Y_boundary, Z_boundary, alpha=0.5, cmap='viridis')
    ax.text(0, 0, 30, 'Dominio válido (región arriba de la superficie)', color='blue')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 30])

    plt.show()


def problema_2_hemisferio():
    """
    f(x,y) = sqrt(1 - (x - 2)^2 - (y + 1)^2)
    Dominio: (x - 2)^2 + (y + 1)^2 <= 1 (círculo centrado en (2,-1) radio 1)
    Muestra: plot 2D del dominio y plot 3D de la superficie (hemisferio).
    """
    print("\nProblema 2: f(x, y) = sqrt(1 - (x - 2)^2 - (y + 1)^2)")
    x, y = symbols('x y')
    f = sqrt(1 - (x - 2)**2 - (y + 1)**2)

    print("La función está definida cuando 1 - (x - 2)^2 - (y + 1)^2 >= 0.")
    print("Dominio: (x - 2)^2 + (y + 1)^2 <= 1 (interior y frontera del círculo).")

    fig = plt.figure(figsize=(12, 6))

    # 2D Domain Plot
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dominio de f(x, y) = sqrt(1 - (x - 2)^2 - (y + 1)^2)')
    ax1.grid(True)

    x_2d = np.linspace(0, 4, 200)
    y_2d = np.linspace(-3, 1, 200)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
    domain_condition = 1 - (X_2d - 2)**2 - (Y_2d + 1)**2 >= 0

    X_2d_valid = np.where(domain_condition, X_2d, np.nan)
    Y_2d_valid = np.where(domain_condition, Y_2d, np.nan)

    ax1.scatter(X_2d_valid, Y_2d_valid, color='skyblue', s=1, label='Dominio válido')

    theta = np.linspace(0, 2*np.pi, 100)
    center_x = 2
    center_y = -1
    radius = 1
    x_boundary = center_x + radius * np.cos(theta)
    y_boundary = center_y + radius * np.sin(theta)
    ax1.plot(x_boundary, y_boundary, color='red', linestyle='--', label='Frontera ((x - 2)^2 + (y + 1)^2 = 1)')

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(-3, 1)
    ax1.legend()

    # 3D Surface Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Superficie f(x, y) = sqrt(1 - (x - 2)^2 - (y + 1)^2)')

    x_surf = np.linspace(1, 3, 50)
    y_surf = np.linspace(-2, 0, 50)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    z_surf = np.where(1 - (x_surf - 2)**2 - (y_surf + 1)**2 >= 0,
                      np.sqrt(1 - (x_surf - 2)**2 - (y_surf + 1)**2),
                      np.nan)

    ax2.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', rstride=1, cstride=1, antialiased=True)
    ax2.set_zlim(0, 1)

    plt.tight_layout()
    plt.show()


def problema_3_log_xyz():
    """
    f(x,y,z) = ln(x*y*z)
    Dominio: x*y*z > 0 -> unión de octantes donde el producto es positivo.
    """
    print("\nProblema 3: f(x, y, z) = ln(x * y * z)")
    x, y, z = symbols('x y z')
    f = log(x * y * z)

    print("La función está definida cuando x * y * z > 0.")
    print("Dominio: unión de las regiones (octantes) donde el producto x*y*z es positivo.")
    print("Por ejemplo: (x>0,y>0,z>0) o (x<0,y<0,z>0) o (x<0,y>0,z<0) o (x>0,y<0,z<0).")

    # Visualización simple indicando octantes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Dominio de f(x, y, z) = ln(x * y * z)')

    # Dibujar planos (proyecciones) como referencia
    ax.plot([-10, 10], [0, 0], [0, 0], color='gray', linestyle='--', alpha=0.5)
    ax.plot([0, 0], [-10, 10], [0, 0], color='gray', linestyle='--', alpha=0.5)
    ax.plot([0, 0], [0, 0], [-10, 10], color='gray', linestyle='--', alpha=0.5)

    ax.text(5, 5, 5, 'xyz > 0', color='blue')
    ax.text(-5, -5, 5, 'xyz > 0', color='blue')
    ax.text(-5, 5, -5, 'xyz > 0', color='blue')
    ax.text(5, -5, -5, 'xyz > 0', color='blue')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()


def problema_4_sin_xy():
    """
    f(x,y) = sin(x*y)
    Dominio: todo R^2
    Representación 2D (sombreado del plano) y superficie 3D de sin(xy).
    """
    print("\nProblema 4: f(x, y) = sin(x * y)")
    x, y = symbols('x y')
    f = sin(x * y)

    print("La función sin(xy) está definida para todos los reales x, y.")
    print("Dominio: R^2.")

    fig = plt.figure(figsize=(12, 6))

    # 2D Domain Plot
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dominio de f(x, y) = sin(xy)')
    ax1.grid(True)
    ax1.add_patch(plt.Rectangle((-10, -10), 20, 20, color='skyblue', alpha=0.5, label='Dominio válido (R^2)'))
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()

    # 3D Surface Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Superficie f(x, y) = sin(xy)')

    x_surf = np.linspace(-5, 5, 100)
    y_surf = np.linspace(-5, 5, 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = np.sin(x_surf * y_surf)

    ax2.plot_surface(x_surf, y_surf, z_surf, cmap='viridis', rstride=5, cstride=5, antialiased=True)
    ax2.set_zlim(-1, 1*

