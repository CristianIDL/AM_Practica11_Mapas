import random
import numpy as np
#el camino es vacio, las paredes son #, las x son lugares donde no puede pasar, el % es para definir el final del  laberinto
#el tesoro para puntos es $, el pozo es @
  
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodios = 2000
acciones = [0, 1, 2, 3]


def generar_laberinto(filas, columnas):

    # Inicializar matriz llena de paredes
    laberinto = [["#" for _ in range(columnas)] for _ in range(filas)]

    # Movimientos posibles (arriba, abajo, izquierda, derecha)
    direcciones = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def dfs(x, y):
        laberinto[x][y] = " "  # Celda libre
        random.shuffle(direcciones)  # Aleatorizar direcciones
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if 1 <= nx < filas - 1 and 1 <= ny < columnas - 1:
                if laberinto[nx][ny] == "#" or laberinto[nx][ny] == "@":
                    # Romper pared intermedia
                    laberinto[x + dx // 2][y + dy // 2] = " "
                    dfs(nx, ny)

    # Iniciar desde (1,1)
    dfs(1, 1)

    laberinto[1][1] = "s"  # Entrada
    fil, col = random.randint(2,9) ,  random.randint(2,9)
    while laberinto[fil][col] == " ":
        fil, col = random.randint(2,9) ,  random.randint(2,9)
    laberinto[fil][col] = "g"  # Salida

    return laberinto

def crear_recompensas(lab,n,m):
    recompensas = [[0 for _ in range(m)]for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if lab[i][j] == '#':
                recompensas[i][j] = -10
            elif lab[i][j] == 'x':
                recompensas[i][j] = -5
            elif lab[i][j] == 'g':
                recompensas[i][j] = 50
            else: 
                recompensas[i][j] = -1
    return recompensas

def state_to_index(x, y, m):
    return x * m + y

def mover(x, y, accion):
    if accion == 0:   # arriba
        return x-1, y
    elif accion == 1: # abajo
        return x+1, y
    elif accion == 2: # izquierda
        return x, y-1
    elif accion == 3: # derecha
        return x, y+1


def entrenar_qlearning(lab, recompensas, n, m):
    Q = np.zeros((n*m, 4))

    # encontrar inicio
    for i in range(n):
        for j in range(m):
            if lab[i][j] == 's':
                inicio = (i, j)

    for ep in range(episodios):
        x, y = inicio
        terminado = False
        total = 0
        while not terminado:
                
            s = state_to_index(x, y, m)

            # politica greedy
            if random.random() < epsilon:
                a = random.choice(acciones)
            else:
                a = np.argmax(Q[s])

            nx, ny = mover(x, y, a)

            # fuera del laberinto
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                recompensa = -10
                total = -10
                nx, ny = inicio
            else:
                celda = lab[nx][ny]
                recompensa = recompensas[nx][ny]
                total += recompensas[nx][ny]
                if celda == '#':
                    nx, ny = inicio
                elif celda == 'g':
                    terminado = True

            s2 = state_to_index(nx, ny, m)

            # actualizacion Q-learning
            Q[s, a] = Q[s, a] + alpha * (
                recompensa + gamma * np.max(Q[s2]) - Q[s, a]
            )

            x, y = nx, ny

    return Q, total

def ejecutar_agente(lab, Q, n, m):
    for i in range(n):
        for j in range(m):
            if lab[i][j] == 's':
                x, y = i, j

    camino = [(x, y)]
    terminado = False
    pasos = 0

    while not terminado and pasos < 200:
        s = state_to_index(x, y, m)
        a = np.argmax(Q[s])
        x, y = mover(x, y, a)

        camino.append((x, y))
        pasos += 1

        if lab[x][y] == 'g':
            terminado = True

    return camino


if __name__ == "__main__":
    n, m = 11,11
    lab = generar_laberinto(n,m)
    recompensas = crear_recompensas(lab,n,m)
    for i in lab:
        print(i)

    Q,total = entrenar_qlearning(lab, recompensas, n, m)
    camino = ejecutar_agente(lab, Q, n, m)
    print(f"Puntaje del agente : {total} ")
    print("Camino aprendido:")
    print(camino)
    for i in range(1,(len(camino)-1)):
        x,y = camino[i]
        lab[x][y] = 'c'

    for i in lab:
        print(i)



