import numpy as np
import cv2 as cv
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

# Variables globales para OpenGL
window_width, window_height = 800, 600
rotation = [0, 0, 0]  
translation = [0, 0, -5]
scale = 1.0  

lkparm = dict(winSize=(15, 15), maxLevel=2,
              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def init_gl():
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window_width / window_height, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def draw_torus():
    """Dibuja un toroide (donut) verde claro"""
    sides = 40
    rings = 20
    inner_radius = 0.5
    outer_radius = 1.0

    glPushMatrix()
    glColor3f(0.6, 1.0, 0.6)  # Verde claro
    for i in range(sides):
        theta1 = i * 2 * np.pi / sides
        theta2 = (i + 1) * 2 * np.pi / sides

        glBegin(GL_QUAD_STRIP)
        for j in range(rings + 1):
            phi = j * 2 * np.pi / rings

            for theta in (theta1, theta2):
                x = (outer_radius + inner_radius * np.cos(phi)) * np.cos(theta)
                y = (outer_radius + inner_radius * np.cos(phi)) * np.sin(theta)
                z = inner_radius * np.sin(phi)
                glVertex3f(x, y, z)
        glEnd()
    glPopMatrix()

def render():
    global translation, rotation, scale
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(*translation)
    glScalef(scale, scale, scale)
    glRotatef(rotation[0], 1.0, 0.0, 0.0)
    glRotatef(rotation[1], 0.0, 1.0, 0.0)
    glRotatef(rotation[2], 0.0, 0.0, 1.0)

    draw_torus()
    glfw.swap_buffers(window)

def update_scale_based_on_direction():
    global scale, translation
    if translation[0] < 0: 
        scale += 0.01
        scale = min(scale, 3.0)  
    elif translation[0] > 0:  
        scale -= 0.01
        scale = max(scale, 0.5)  

def update_transformations(p1, p0):
    global translation, rotation
    delta = (p1 - p0).reshape(-1, 2)  # Garantiza forma (n, 2)
    mean_delta = np.mean(delta, axis=0)

    # Actualizar traslación y rotación
    translation[0] += mean_delta[0] * 0.01  # Movimiento horizontal
    translation[1] -= mean_delta[1] * 0.01  # Movimiento vertical
    rotation[1] += mean_delta[0] * 0.5  # Rotación sobre Y
    rotation[0] -= mean_delta[1] * 0.5  # Rotación sobre X

    # Actualizar escala en función de la dirección horizontal
    update_scale_based_on_direction()

# Bucle principal
def main():
    global window

    # Iniciar GLFW
    if not glfw.init():
        raise Exception("No se pudo inicializar GLFW")

    window = glfw.create_window(window_width, window_height, "Figura 3D con GLFW", None, None)
    if not window:
        glfw.terminate()
        raise Exception("No se pudo crear la ventana GLFW")

    glfw.make_context_current(window)
    init_gl()

    # Iniciar la captura de video
    cap = cv.VideoCapture(0)
    _, vframe = cap.read()
    vframe = cv.flip(vframe, 1)  # Invertir horizontalmente
    vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)

    # Matriz inicial de puntos 7x7 más centrada
    rows, cols = vgris.shape
    step = 30
    p0 = np.array([[x, y] for y in range(rows // 2 - 90, rows // 2 + 90, step)
                   for x in range(cols // 2 - 90, cols // 2 + 90, step)])
    p0 = np.float32(p0[:, np.newaxis, :])

    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # Invertir horizontalmente
        fgris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calcular flujo óptico
        p1, st, err = cv.calcOpticalFlowPyrLK(vgris, fgris, p0, None, **lkparm)

        if p1 is not None:
            bp1 = p1[st == 1]
            bp0 = p0[st == 1]

            # Dibujar puntos cuadrados
            for nv in bp1:
                a, b = (int(x) for x in nv.ravel())
                top_left = (a - 3, b - 3)
                bottom_right = (a + 3, b + 3)
                frame = cv.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)

            # Actualizar transformaciones de OpenGL
            update_transformations(bp1, bp0)

        cv.imshow('Video', frame)
        render()
        vgris = fgris.copy()

        if cv.waitKey(1) & 0xFF == 27:
            break

        glfw.poll_events()

    glfw.terminate()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
