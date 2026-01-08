"""
Módulo para generar mapas de laberintos de forma pseudoaleatoria.
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import random
import numpy as np
from typing import Tuple, List, Optional
from src.config_manager import ConfigManager


class MapGenerator:
    """
    Genera mapas de laberintos 12x12 con elementos pseudoaleatorios.
    """
    
    def __init__(self, config: ConfigManager, seed: Optional[int] = None):
        """
        Inicializa el generador de mapas.
        
        Args:
            config: Instancia del ConfigManager
            seed: Semilla para generación pseudoaleatoria (opcional)
        """
        self.config = config
        self.map_size = 12
        self.inner_size = 10
        self.map_grid = None
        
        # Establecer semilla si se proporciona
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 999999)
            random.seed(self.seed)
            np.random.seed(self.seed)
    
    def generate_map(self, 
                     wall_density: float = 0.2,
                     treasure_count: int = 3,
                     pit_count: int = 3) -> np.ndarray:
        """
        Genera un mapa completo del laberinto.
        
        Args:
            wall_density: Densidad de paredes internas (0.0 a 1.0)
            treasure_count: Número de tesoros a colocar
            pit_count: Número de pozos a colocar
            
        Returns:
            Matriz numpy con el mapa generado
        """
        # Tarea 2.1: Crear estructura básica
        self._create_base_structure()
        
        # Tarea 2.2: Colocar elementos fijos (S y G)
        start_pos, goal_pos = self._place_start_and_goal()
        
        # Tarea 2.3: Generar elementos pseudoaleatorios
        self._place_internal_walls(wall_density, [start_pos, goal_pos])
        self._place_special_items(treasure_count, pit_count, [start_pos, goal_pos])
        
        return self.map_grid
    
    def _create_base_structure(self):
        """
        Tarea 2.1: Crea la estructura base del mapa 12x12.
        - Bordes con paredes
        - Interior con caminos libres
        """
        # Inicializar matriz con caminos libres
        path_char = self.config.get_char('PATH')
        self.map_grid = np.full((self.map_size, self.map_size), path_char, dtype=str)
        
        # Establecer bordes como paredes
        wall_char = self.config.get_char('WALL')
        
        # Bordes superior e inferior
        self.map_grid[0, :] = wall_char
        self.map_grid[11, :] = wall_char
        
        # Bordes izquierdo y derecho
        self.map_grid[:, 0] = wall_char
        self.map_grid[:, 11] = wall_char
    
    def _place_start_and_goal(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Tarea 2.2: Coloca S (Start) y G (Goal) en posiciones válidas.
        
        Returns:
            Tupla con las posiciones (start, goal)
        """
        start_char = self.config.get_char('START')
        goal_char = self.config.get_char('GOAL')
        
        # Estrategia: S en la parte inferior, G en la parte superior
        # S siempre en la fila 11 (borde inferior)
        # G siempre en la fila 0 (borde superior)
        
        # Colocar START en el borde inferior (fila 11)
        start_col = 1
        '''start_col = random.randint(1, 10)  # Columnas del 1 al 10'''
        start_pos = (11, start_col)
        self.map_grid[start_pos] = start_char
        
        # Colocar GOAL en el borde superior (fila 0)
        goal_col = 10
        '''goal_col = random.randint(1, 10)'''
        goal_pos = (0, goal_col)
        self.map_grid[goal_pos] = goal_char
        
        return start_pos, goal_pos
    
    def _place_internal_walls(self, density: float, reserved_positions: List[Tuple[int, int]]):
        """
        Tarea 2.3: Coloca paredes internas de forma pseudoaleatoria.
        
        Args:
            density: Proporción de paredes (0.0 a 1.0)
            reserved_positions: Posiciones que no deben ser modificadas
        """
        wall_char = self.config.get_char('WALL')
        
        # Calcular número de paredes a colocar
        total_inner_cells = self.inner_size * self.inner_size
        num_walls = int(total_inner_cells * density)
        
        placed_walls = 0
        attempts = 0
        max_attempts = num_walls * 10  # Evitar bucles infinitos
        
        while placed_walls < num_walls and attempts < max_attempts:
            # Generar posición aleatoria en el interior
            row = random.randint(1, 10)
            col = random.randint(1, 10)
            pos = (row, col)
            
            # Verificar que no esté en posiciones reservadas
            if pos not in reserved_positions:
                # Verificar que sea un camino libre
                if self.map_grid[pos] == self.config.get_char('PATH'):
                    self.map_grid[pos] = wall_char
                    placed_walls += 1
            
            attempts += 1
    
    def _place_special_items(self, 
                            treasure_count: int, 
                            pit_count: int,
                            reserved_positions: List[Tuple[int, int]]):
        """
        Tarea 2.3: Coloca tesoros y pozos de forma pseudoaleatoria.
        
        Args:
            treasure_count: Número de tesoros
            pit_count: Número de pozos
            reserved_positions: Posiciones que no deben ser modificadas
        """
        treasure_char = self.config.get_char('TREASURE')
        pit_char = self.config.get_char('PIT')
        path_char = self.config.get_char('PATH')
        
        # Colocar tesoros
        placed_treasures = 0
        attempts = 0
        max_attempts = treasure_count * 20
        
        while placed_treasures < treasure_count and attempts < max_attempts:
            row = random.randint(1, 10)
            col = random.randint(1, 10)
            pos = (row, col)
            
            if pos not in reserved_positions and self.map_grid[pos] == path_char:
                self.map_grid[pos] = treasure_char
                reserved_positions.append(pos)
                placed_treasures += 1
            
            attempts += 1
        
        # Colocar pozos
        placed_pits = 0
        attempts = 0
        
        while placed_pits < pit_count and attempts < max_attempts:
            row = random.randint(1, 10)
            col = random.randint(1, 10)
            pos = (row, col)
            
            if pos not in reserved_positions and self.map_grid[pos] == path_char:
                self.map_grid[pos] = pit_char
                reserved_positions.append(pos)
                placed_pits += 1
            
            attempts += 1
    
    def get_start_position(self) -> Optional[Tuple[int, int]]:
        """
        Encuentra la posición del START en el mapa.
        
        Returns:
            Tupla (fila, columna) o None si no se encuentra
        """
        start_char = self.config.get_char('START')
        positions = np.where(self.map_grid == start_char)
        
        if len(positions[0]) > 0:
            return (positions[0][0], positions[1][0])
        return None
    
    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        """
        Encuentra la posición del GOAL en el mapa.
        
        Returns:
            Tupla (fila, columna) o None si no se encuentra
        """
        goal_char = self.config.get_char('GOAL')
        positions = np.where(self.map_grid == goal_char)
        
        if len(positions[0]) > 0:
            return (positions[0][0], positions[1][0])
        return None
    
    def print_map(self):
        """Imprime el mapa en consola de forma legible."""
        if self.map_grid is None:
            print("No hay mapa generado aún.")
            return
        
        print("\n" + "=" * 40)
        print(f"  MAPA GENERADO (Semilla: {self.seed})")
        print("=" * 40)
        
        for row in self.map_grid:
            print(''.join(row))
        
        print("=" * 40)
        
        # Mostrar información adicional
        start_pos = self.get_start_position()
        goal_pos = self.get_goal_position()
        
        print(f"\nInformación:")
        print(f"  Start (S): {start_pos}")
        print(f"  Goal (G):  {goal_pos}")
        print(f"  Semilla:   {self.seed}")
    
    def save_map(self, filename: str):
        """
        Guarda el mapa en un archivo de texto.
        
        Args:
            filename: Nombre del archivo (con ruta)
        """
        import os
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for row in self.map_grid:
                f.write(''.join(row) + '\n')
        
        print(f"Mapa guardado en: {filename}")


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar configuración
    config = ConfigManager('../config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración")
        exit(1)
    
    print("\n--- Generando mapas de ejemplo ---\n")
    
    # Generar varios mapas con diferentes configuraciones
    for i in range(3):
        print(f"\n--- Mapa {i+1} ---")
        generator = MapGenerator(config, seed=i*100)
        
        # Generar con diferentes densidades
        density = 0.1 + (i * 0.1)
        generator.generate_map(
            wall_density=density,
            treasure_count=2 + i,
            pit_count=2 + i
        )
        
        generator.print_map()
        
        # Guardar el mapa
        generator.save_map(f'../maps/generated/mapa_{i+1:03d}.txt')