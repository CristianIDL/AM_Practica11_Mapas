import random
import numpy as np
from collections import deque

#el camino es vacio, las paredes son #, las x son lugares donde no puede pasar, el % es para definir el final del  laberinto
#el tesoro para puntos es $, el pozo es @

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodios = 2000
acciones = [0, 1, 2, 3]
max_pasos = 300


def vecinos_validos(lab, x, y, n, m):
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < n and 0 <= ny < m:
            if lab[nx][ny] not in ['#', '@']:
                yield nx, ny


def bfs_camino(lab, inicio, objetivo, n, m):
    cola = deque([inicio])
    padres = {inicio: None}

    while cola:
        actual = cola.popleft()
        if actual == objetivo:
            break

        for v in vecinos_validos(lab, actual[0], actual[1], n, m):
            if v not in padres:
                padres[v] = actual
                cola.append(v)

    if objetivo not in padres:
        return None  # No hay camino

    # reconstruir camino
    camino = []
    nodo = objetivo
    while nodo:
        camino.append(nodo)
        nodo = padres[nodo]
    return camino[::-1]


def camino_optimo_con_tesoros(lab, n, m):
    tesoros = []
    for i in range(n):
        for j in range(m):
            if lab[i][j] == 's':
                inicio = (i, j)
            elif lab[i][j] == 'g':
                meta = (i, j)
            elif lab[i][j] == '$':
                tesoros.append((i, j))

    camino_total = []
    actual = inicio

    for t in tesoros:
        tramo = bfs_camino(lab, actual, t, n, m)
        if tramo is None:
            continue
        camino_total += tramo[:-1]
        actual = t

    tramo_final = bfs_camino(lab, actual, meta, n, m)
    if tramo_final:
        camino_total += tramo_final

    return camino_total

def generar_laberinto(filas, columnas):

    # Inicializar matriz llena de paredes
    laberinto = [["#" for _ in range(columnas)] for _ in range(filas)]

    # Movimientos posibles (arriba, abajo, izquierda, derecha)
    direcciones = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    objetos = ["$","x","@"]

    for i in range(filas):
        fil,col = random.randint(2,filas-2), random.randint(2,columnas-2)
        if random.choice(objetos) == "$":
            laberinto[col][fil] = "$"
        elif random.choice(objetos) == "x":
            laberinto[col][fil] = "x"
        else:
            laberinto[col][fil] = "@"

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

def crear_recompensas(lab, n, m):
    recompensas = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if lab[i][j] == '#':
                recompensas[i][j] = -15
            elif lab[i][j] == 'x':
                recompensas[i][j] = -10
            elif lab[i][j] == '@': 
                recompensas[i][j] = -100
            elif lab[i][j] == '$': 
                recompensas[i][j] = 10  
            elif lab[i][j] == 'g': 
                recompensas[i][j] = 500 
            else: 
                recompensas[i][j] = -2  
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
    max_pasos = 300

    for i in range(n):
        for j in range(m):
            if lab[i][j] == 's':
                inicio = (i, j)

    epsilon = 1.0
    epsilon_min = 0.05
    decay = 0.995

    for ep in range(episodios):
        x, y = inicio
        terminado = False
        pasos = 0
        total = 0

        while not terminado and pasos < max_pasos:
            pasos += 1
            s = state_to_index(x, y, m)


            if random.random() < epsilon:
                a = random.choice(acciones)
            else:
                a = np.argmax(Q[s])

            nx, ny = mover(x, y, a)

            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                recompensa = -10
                target_q = recompensa # Penalizar y no mover
                Q[s, a] += alpha * (target_q - Q[s, a])
                continue 
            celda = lab[nx][ny]
            recompensa = recompensas[nx][ny]

 
            if celda == '#':

                Q[s, a] += alpha * (recompensa + gamma * np.max(Q[s]) - Q[s, a])
                continue 

            if celda == 'g': # Meta
                terminado = True
            elif celda == '@': # Pozo/Muerte
                terminado = True
            
            s2 = state_to_index(nx, ny, m)
            if terminado:
                Q[s, a] += alpha * (recompensa - Q[s, a])
            else:
                Q[s, a] += alpha * (recompensa + gamma * np.max(Q[s2]) - Q[s, a])

            x, y = nx, ny

        epsilon = max(epsilon_min, epsilon * decay)

    return Q, total

def ejecutar_agente(lab, Q, n, m):
    for i in range(n):
        for j in range(m):
            if lab[i][j] == 's':
                x, y = i, j
    camino = [(x, y)]
    visitados = [(x, y)] 
    pasos = 0

    while pasos < 100:
        s = state_to_index(x, y, m)
        a = np.argmax(Q[s])
        nx, ny = mover(x, y, a)

        if nx < 0 or nx >= n or ny < 0 or ny >= m or lab[nx][ny] == '#':
            break 

        if len(camino) > 4 and camino[-1] == camino[-3] and camino[-2] == (nx, ny):
             print("¡Bucle detectado en ejecución!")
             break

        camino.append((nx, ny))
        x, y = nx, ny
        pasos += 1

        if lab[nx][ny] == 'g':
            print("¡Meta alcanzada!")
            break

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

    camino_simulado = camino_optimo_con_tesoros(lab, n, m)

    print("Camino óptimo simulado:")
    print(camino_simulado)

    for x, y in camino_simulado:
        if lab[x][y] == " ":
            lab[x][y] = "c"

    for fila in lab:
        print(fila)
        

