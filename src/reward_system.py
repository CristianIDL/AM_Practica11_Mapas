"""
Módulo para gestionar sistemas de recompensas de los laberintos.
Implementa las reglas específicas para Lab1 (simple) y Lab2 (tesoros).
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import numpy as np
from typing import Tuple, Optional
from src.config_manager import ConfigManager


class RewardSystemBase:
    """
    Clase base para sistemas de recompensas.
    Define la interfaz común para ambos laberintos.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Inicializa el sistema de recompensas.
        
        Args:
            config: Instancia del ConfigManager
        """
        self.config = config
        self.reward_matrix = None
    
    def create_reward_matrix(self, map_grid: np.ndarray) -> np.ndarray:
        """
        Crea la matriz de recompensas basada en el mapa.
        Debe ser implementado por las subclases.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Matriz de recompensas (mismo tamaño que map_grid)
        """
        raise NotImplementedError("Subclases deben implementar create_reward_matrix")
    
    def get_reward(self, map_grid: np.ndarray, position: Tuple[int, int]) -> float:
        """
        Obtiene la recompensa para una posición específica.
        
        Args:
            map_grid: Matriz del mapa
            position: Tupla (fila, columna)
            
        Returns:
            Valor de recompensa
        """
        if self.reward_matrix is None:
            self.reward_matrix = self.create_reward_matrix(map_grid)
        
        row, col = position
        return self.reward_matrix[row, col]


class RewardSystemLab1(RewardSystemBase):
    """
    Sistema de recompensas para Laberinto 1 (Decisiones Estocásticas).
    
    Reglas:
    - -1 por cada paso
    - -10 si choca con paredes
    - +50 si llega al objetivo
    """
    
    # Constantes de recompensa
    STEP_PENALTY = -1
    WALL_PENALTY = -10
    GOAL_REWARD = 50
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.episode_steps = 0  # Contador de pasos en el episodio actual
    
    def create_reward_matrix(self, map_grid: np.ndarray) -> np.ndarray:
        """
        Crea la matriz de recompensas para Lab1.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Matriz de recompensas
        """
        rows, cols = map_grid.shape
        rewards = np.zeros((rows, cols), dtype=float)
        
        wall_char = self.config.get_char('WALL')
        goal_char = self.config.get_char('GOAL')
        
        for i in range(rows):
            for j in range(cols):
                cell = map_grid[i, j]
                
                if cell == wall_char:
                    rewards[i, j] = self.WALL_PENALTY
                elif cell == goal_char:
                    rewards[i, j] = self.GOAL_REWARD
                else:
                    # Camino libre, START, o cualquier otra casilla transitable
                    rewards[i, j] = self.STEP_PENALTY
        
        self.reward_matrix = rewards
        return rewards
    
    def get_step_reward(self, 
                       map_grid: np.ndarray, 
                       current_pos: Tuple[int, int],
                       action: int,
                       next_pos: Tuple[int, int],
                       out_of_bounds: bool = False) -> Tuple[float, bool]:
        """
        Calcula la recompensa por un paso específico.
        
        Args:
            map_grid: Matriz del mapa
            current_pos: Posición actual
            action: Acción tomada (0=arriba, 1=abajo, 2=izq, 3=der)
            next_pos: Posición siguiente
            out_of_bounds: Si True, el agente intentó salir del mapa
            
        Returns:
            Tupla (recompensa, episodio_terminado)
        """
        # Fuera de límites
        if out_of_bounds:
            return self.WALL_PENALTY, False
        
        # Obtener recompensa de la celda
        reward = self.get_reward(map_grid, next_pos)
        
        # Verificar si llegó a la meta
        goal_char = self.config.get_char('GOAL')
        terminated = (map_grid[next_pos] == goal_char)
        
        # Si chocó con pared, no se mueve (permanece en current_pos)
        wall_char = self.config.get_char('WALL')
        if map_grid[next_pos] == wall_char:
            terminated = False
        
        self.episode_steps += 1
        
        return reward, terminated
    
    def reset_episode(self):
        """Reinicia el contador de pasos para un nuevo episodio."""
        self.episode_steps = 0
    
    def get_episode_score(self) -> int:
        """
        Calcula el puntaje total del episodio.
        
        Returns:
            Puntaje acumulado
        """
        # En Lab1, el score es simplemente la suma de recompensas
        # (ya contabilizado durante el entrenamiento)
        return -self.episode_steps  # Penalización por pasos
    
    def print_config(self):
        """Imprime la configuración de recompensas."""
        print("\n" + "="*50)
        print("  CONFIGURACIÓN LAB1 (Simple)")
        print("="*50)
        print(f"  Paso normal:      {self.STEP_PENALTY:+.0f} puntos")
        print(f"  Choque con pared: {self.WALL_PENALTY:+.0f} puntos")
        print(f"  Llegar a meta:    {self.GOAL_REWARD:+.0f} puntos")
        print("="*50)


class RewardSystemLab2(RewardSystemBase):
    """
    Sistema de recompensas para Laberinto 2 (Con Tesoros).
    
    Reglas:
    - -1 por cada paso
    - +20 por recoger tesoro
    - -20 por caer en pozo
    - +50 al llegar a la meta
    - -10 si choca con paredes
    - Bonificación por tiempo:
        ≤40 pasos:  +30 puntos
        ≤70 pasos:  +20 puntos
        ≤100 pasos: +10 puntos
    """
    
    # Constantes de recompensa
    STEP_PENALTY = -1
    TREASURE_REWARD = 20
    PIT_PENALTY = -20
    GOAL_REWARD = 100      # Aumentado de 50 a 100
    WALL_PENALTY = -10
    
    # Bonificaciones por tiempo
    TIME_BONUS_THRESHOLD_1 = 40
    TIME_BONUS_VALUE_1 = 30
    
    TIME_BONUS_THRESHOLD_2 = 70
    TIME_BONUS_VALUE_2 = 20
    
    TIME_BONUS_THRESHOLD_3 = 100
    TIME_BONUS_VALUE_3 = 10
    
    def __init__(self, config: ConfigManager):
        super().__init__(config)
        self.episode_steps = 0
        self.treasures_collected = 0
        self.total_treasures = 0
    
    def create_reward_matrix(self, map_grid: np.ndarray) -> np.ndarray:
        """
        Crea la matriz de recompensas para Lab2.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Matriz de recompensas
        """
        rows, cols = map_grid.shape
        rewards = np.zeros((rows, cols), dtype=float)
        
        wall_char = self.config.get_char('WALL')
        goal_char = self.config.get_char('GOAL')
        treasure_char = self.config.get_char('TREASURE')
        pit_char = self.config.get_char('PIT')
        
        treasure_count = 0
        
        for i in range(rows):
            for j in range(cols):
                cell = map_grid[i, j]
                
                if cell == wall_char:
                    rewards[i, j] = self.WALL_PENALTY
                elif cell == goal_char:
                    rewards[i, j] = self.GOAL_REWARD
                elif cell == treasure_char:
                    rewards[i, j] = self.TREASURE_REWARD
                    treasure_count += 1
                elif cell == pit_char:
                    rewards[i, j] = self.PIT_PENALTY
                else:
                    # Camino libre o START
                    rewards[i, j] = self.STEP_PENALTY
        
        self.reward_matrix = rewards
        self.total_treasures = treasure_count
        return rewards
    
    def get_step_reward(self, 
                       map_grid: np.ndarray, 
                       current_pos: Tuple[int, int],
                       action: int,
                       next_pos: Tuple[int, int],
                       out_of_bounds: bool = False) -> Tuple[float, bool]:
        """
        Calcula la recompensa por un paso específico en Lab2.
        
        Args:
            map_grid: Matriz del mapa
            current_pos: Posición actual
            action: Acción tomada
            next_pos: Posición siguiente
            out_of_bounds: Si el agente salió del mapa
            
        Returns:
            Tupla (recompensa, episodio_terminado)
        """
        # Fuera de límites
        if out_of_bounds:
            return self.WALL_PENALTY, False
        
        # Obtener caracteres
        goal_char = self.config.get_char('GOAL')
        pit_char = self.config.get_char('PIT')
        wall_char = self.config.get_char('WALL')
        treasure_char = self.config.get_char('TREASURE')
        
        cell = map_grid[next_pos]
        terminated = False
        
        # Verificar tipo de celda y obtener recompensa
        if cell == wall_char:
            # Chocó con pared, no se mueve
            reward = self.WALL_PENALTY
            terminated = False
        
        elif cell == goal_char:
            # Llegó a la meta
            reward = self.GOAL_REWARD
            terminated = True
        
        elif cell == pit_char:
            # Cayó en un pozo - episodio termina
            reward = self.PIT_PENALTY
            terminated = True
        
        elif cell == treasure_char:
            # Recogió un tesoro
            reward = self.TREASURE_REWARD
            self.treasures_collected += 1
        
        else:
            # Camino normal (incluye START y PATH)
            reward = self.STEP_PENALTY
        
        self.episode_steps += 1
        
        return reward, terminated
    
    def calculate_time_bonus(self, steps: int) -> int:
        """
        Calcula la bonificación por tiempo según los pasos dados.
        
        Args:
            steps: Número de pasos del episodio
            
        Returns:
            Bonificación (0, 10, 20, o 30)
        """
        if steps <= self.TIME_BONUS_THRESHOLD_1:
            return self.TIME_BONUS_VALUE_1
        elif steps <= self.TIME_BONUS_THRESHOLD_2:
            return self.TIME_BONUS_VALUE_2
        elif steps <= self.TIME_BONUS_THRESHOLD_3:
            return self.TIME_BONUS_VALUE_3
        else:
            return 0
    
    def get_final_reward(self, reached_goal: bool) -> float:
        """
        Calcula la recompensa final del episodio (incluye bonificación por tiempo).
        
        Args:
            reached_goal: Si el agente alcanzó la meta
            
        Returns:
            Recompensa final adicional
        """
        if not reached_goal:
            return 0
        
        # Bonificación por tiempo solo si llegó a la meta
        time_bonus = self.calculate_time_bonus(self.episode_steps)
        
        return time_bonus
    
    def reset_episode(self):
        """Reinicia contadores para un nuevo episodio."""
        self.episode_steps = 0
        self.treasures_collected = 0
    
    def get_episode_score(self, reached_goal: bool = False) -> dict:
        """
        Calcula estadísticas completas del episodio.
        
        Args:
            reached_goal: Si el agente llegó a la meta
            
        Returns:
            Diccionario con estadísticas del episodio
        """
        time_bonus = self.calculate_time_bonus(self.episode_steps) if reached_goal else 0
        
        return {
            'steps': self.episode_steps,
            'treasures_collected': self.treasures_collected,
            'total_treasures': self.total_treasures,
            'time_bonus': time_bonus,
            'reached_goal': reached_goal
        }
    
    def print_config(self):
        """Imprime la configuración de recompensas."""
        print("\n" + "="*50)
        print("  CONFIGURACIÓN LAB2 (Tesoros)")
        print("="*50)
        print(f"  Paso normal:      {self.STEP_PENALTY:+.0f} puntos")
        print(f"  Tesoro:           {self.TREASURE_REWARD:+.0f} puntos")
        print(f"  Pozo:             {self.PIT_PENALTY:+.0f} puntos")
        print(f"  Choque con pared: {self.WALL_PENALTY:+.0f} puntos")
        print(f"  Llegar a meta:    {self.GOAL_REWARD:+.0f} puntos")
        print("\n  Bonificación por Tiempo:")
        print(f"    ≤{self.TIME_BONUS_THRESHOLD_1} pasos:  +{self.TIME_BONUS_VALUE_1} puntos")
        print(f"    ≤{self.TIME_BONUS_THRESHOLD_2} pasos:  +{self.TIME_BONUS_VALUE_2} puntos")
        print(f"    ≤{self.TIME_BONUS_THRESHOLD_3} pasos: +{self.TIME_BONUS_VALUE_3} puntos")
        print("="*50)


# Factory para crear el sistema de recompensas apropiado
def create_reward_system(lab_type: str, config: ConfigManager) -> RewardSystemBase:
    """
    Factory para crear el sistema de recompensas según el tipo de laberinto.
    
    Args:
        lab_type: "lab1" o "lab2"
        config: Instancia del ConfigManager
        
    Returns:
        Instancia del sistema de recompensas apropiado
        
    Raises:
        ValueError: Si lab_type no es válido
    """
    lab_type = lab_type.lower()
    
    if lab_type == "lab1":
        return RewardSystemLab1(config)
    elif lab_type == "lab2":
        return RewardSystemLab2(config)
    else:
        raise ValueError(f"Tipo de laberinto no válido: {lab_type}. Use 'lab1' o 'lab2'")


# Ejemplo de uso
if __name__ == "__main__":
    from src.map_loader import MapLoader
    
    # Cargar configuración
    config = ConfigManager('config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración")
        exit(1)
    
    print("\n" + "="*60)
    print("  PRUEBA DE REWARD SYSTEMS")
    print("="*60)
    
    # Crear mapa de ejemplo para Lab1
    example_lab1 = np.array([
        list("##########G#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#'####'''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#S##########")
    ])
    
    # Crear mapa de ejemplo para Lab2
    example_lab2 = np.array([
        list("##########G#"),
        list("#''''''''''#"),
        list("#''T'''T'''#"),
        list("#''''''''''#"),
        list("#'####''X''#"),
        list("#''''''''''#"),
        list("#''''''''''#"),
        list("#''T'''''''#"),
        list("#''''''''''#"),
        list("#''''''X'''#"),
        list("#''''''''''#"),
        list("#S##########")
    ])
    
    # Prueba Lab1
    print("\n--- SISTEMA LAB1 ---")
    reward_lab1 = create_reward_system("lab1", config)
    reward_lab1.print_config()
    
    print("\nMatriz de recompensas Lab1 (primeras 3 filas):")
    rewards1 = reward_lab1.create_reward_matrix(example_lab1)
    for i in range(3):
        print(f"  Fila {i}: {rewards1[i]}")
    
    # Simular algunos pasos
    print("\nSimulación de pasos:")
    reward_lab1.reset_episode()
    
    # Paso 1: Mover de S hacia arriba
    current = (11, 1)
    next_pos = (10, 1)
    reward, done = reward_lab1.get_step_reward(example_lab1, current, 0, next_pos)
    print(f"  Paso 1: {current} → {next_pos} | Recompensa: {reward:+.0f} | Terminado: {done}")
    
    # Paso 2: Intentar chocar con pared
    current = (10, 1)
    next_pos = (10, 0)  # Pared
    reward, done = reward_lab1.get_step_reward(example_lab1, current, 2, next_pos)
    print(f"  Paso 2: {current} → {next_pos} | Recompensa: {reward:+.0f} | Terminado: {done}")
    
    # Prueba Lab2
    print("\n\n--- SISTEMA LAB2 ---")
    reward_lab2 = create_reward_system("lab2", config)
    reward_lab2.print_config()
    
    print("\nMatriz de recompensas Lab2 (primeras 3 filas):")
    rewards2 = reward_lab2.create_reward_matrix(example_lab2)
    for i in range(3):
        print(f"  Fila {i}: {rewards2[i]}")
    
    print(f"\nTesoros totales en el mapa: {reward_lab2.total_treasures}")
    
    # Simular episodio completo
    print("\nSimulación de episodio:")
    reward_lab2.reset_episode()
    
    # Paso 1: Mover normalmente
    reward, done = reward_lab2.get_step_reward(example_lab2, (11, 1), 0, (10, 1))
    print(f"  Paso 1: Movimiento normal | Recompensa: {reward:+.0f}")
    
    # Paso 2: Recoger tesoro
    reward, done = reward_lab2.get_step_reward(example_lab2, (3, 1), 0, (2, 2))
    print(f"  Paso 2: Recogió tesoro | Recompensa: {reward:+.0f} | Tesoros: {reward_lab2.treasures_collected}")
    
    # Paso 3: Caer en pozo
    reward, done = reward_lab2.get_step_reward(example_lab2, (4, 7), 0, (4, 7))
    print(f"  Paso 3: Cayó en pozo | Recompensa: {reward:+.0f} | Terminado: {done}")
    
    # Calcular bonificación por tiempo
    print("\nBonificación por tiempo:")
    for steps in [35, 60, 85, 110]:
        bonus = reward_lab2.calculate_time_bonus(steps)
        print(f"  {steps:3d} pasos → +{bonus} puntos")
    
    print("\n" + "="*60)
    print("  Pruebas completadas exitosamente")
    print("="*60)