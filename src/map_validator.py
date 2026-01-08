"""
Módulo para validar que los mapas generados sean válidos.
Verifica que exista un camino de START a GOAL.
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Set, Optional
from src.config_manager import ConfigManager

class MapValidator:
    """
    Valida mapas de laberintos usando BFS (Breadth-First Search).
    """
    
    def __init__(self, config: ConfigManager):
        """
        Inicializa el validador de mapas.
        
        Args:
            config: Instancia del ConfigManager
        """
        self.config = config
    
    def is_valid_map(self, map_grid: np.ndarray, verbose: bool = False) -> bool:
        """
        Verifica si el mapa es válido (existe camino de S a G).
        
        Args:
            map_grid: Matriz del mapa
            verbose: Si True, muestra información detallada
            
        Returns:
            True si el mapa es válido, False en caso contrario
        """
        # Encontrar posiciones de START y GOAL
        start_pos = self._find_position(map_grid, 'START')
        goal_pos = self._find_position(map_grid, 'GOAL')
        
        if start_pos is None:
            if verbose:
                print("No se encontró la posición START")
            return False
        
        if goal_pos is None:
            if verbose:
                print("No se encontró la posición GOAL")
            return False
        
        # Verificar si existe un camino
        path_exists, path_length = self._bfs_path_exists(map_grid, start_pos, goal_pos)
        
        if verbose:
            if path_exists:
                print(f"Mapa válido - Existe camino de longitud {path_length}")
            else:
                print("Mapa inválido - No existe camino de S a G")
        
        return path_exists
    
    def _find_position(self, map_grid: np.ndarray, tile_type: str) -> Optional[Tuple[int, int]]:
        """
        Encuentra la posición de un tipo de casilla en el mapa.
        
        Args:
            map_grid: Matriz del mapa
            tile_type: Tipo de casilla a buscar
            
        Returns:
            Tupla (fila, columna) o None si no se encuentra
        """
        char = self.config.get_char(tile_type)
        positions = np.where(map_grid == char)
        
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None
    
    def _bfs_path_exists(self, 
                        map_grid: np.ndarray, 
                        start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> Tuple[bool, int]:
        """
        Implementa BFS para verificar si existe un camino de start a goal.
        
        Args:
            map_grid: Matriz del mapa
            start: Posición inicial
            goal: Posición objetivo
            
        Returns:
            Tupla (existe_camino, longitud_camino)
        """
        # Caracteres transitables
        wall_char = self.config.get_char('WALL')
        
        # Cola para BFS: (posición, distancia)
        queue = deque([(start, 0)])
        visited: Set[Tuple[int, int]] = {start}
        
        # Direcciones: arriba, abajo, izquierda, derecha
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            current_pos, distance = queue.popleft()
            
            # ¿Llegamos al objetivo?
            if current_pos == goal:
                return True, distance
            
            # Explorar vecinos
            for dr, dc in directions:
                new_row = current_pos[0] + dr
                new_col = current_pos[1] + dc
                new_pos = (new_row, new_col)
                
                # Verificar límites del mapa
                if not (0 <= new_row < map_grid.shape[0] and 
                       0 <= new_col < map_grid.shape[1]):
                    continue
                
                # Verificar si ya fue visitado
                if new_pos in visited:
                    continue
                
                # Verificar si es transitable (no es pared)
                if map_grid[new_pos] != wall_char:
                    visited.add(new_pos)
                    queue.append((new_pos, distance + 1))
        
        # No se encontró camino
        return False, -1
    
    def get_path(self, map_grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Obtiene el camino más corto de START a GOAL usando BFS.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Lista de posiciones que forman el camino, o None si no existe
        """
        start_pos = self._find_position(map_grid, 'START')
        goal_pos = self._find_position(map_grid, 'GOAL')
        
        if start_pos is None or goal_pos is None:
            return None
        
        wall_char = self.config.get_char('WALL')
        
        # Cola: (posición, camino)
        queue = deque([(start_pos, [start_pos])])
        visited: Set[Tuple[int, int]] = {start_pos}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            current_pos, path = queue.popleft()
            
            if current_pos == goal_pos:
                return path
            
            for dr, dc in directions:
                new_row = current_pos[0] + dr
                new_col = current_pos[1] + dc
                new_pos = (new_row, new_col)
                
                if not (0 <= new_row < map_grid.shape[0] and 
                       0 <= new_col < map_grid.shape[1]):
                    continue
                
                if new_pos in visited:
                    continue
                
                if map_grid[new_pos] != wall_char:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return None
    
    def visualize_path(self, map_grid: np.ndarray):
        """
        Visualiza el mapa con el camino marcado.
        
        Args:
            map_grid: Matriz del mapa
        """
        path = self.get_path(map_grid)
        
        if path is None:
            print("No existe camino válido para visualizar")
            return
        
        # Crear copia del mapa para visualización
        visual_map = map_grid.copy()
        
        # Marcar el camino (excepto START y GOAL)
        start_char = self.config.get_char('START')
        goal_char = self.config.get_char('GOAL')
        
        for pos in path:
            if visual_map[pos] not in [start_char, goal_char]:
                visual_map[pos] = '·'  # Marcador visual del camino
        
        print("\n" + "=" * 40)
        print("  VISUALIZACIÓN DEL CAMINO")
        print("=" * 40)
        
        for row in visual_map:
            print(''.join(row))
        
        print("=" * 40)
        print(f"Longitud del camino: {len(path)} pasos")
        print("(· = camino óptimo)")
    
    def get_statistics(self, map_grid: np.ndarray) -> dict:
        """
        Obtiene estadísticas del mapa.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'valid': self.is_valid_map(map_grid),
            'size': map_grid.shape,
            'tiles': {}
        }
        
        # Contar cada tipo de casilla
        for tile_type in ['START', 'GOAL', 'WALL', 'PATH', 'TREASURE', 'PIT']:
            char = self.config.get_char(tile_type)
            count = np.sum(map_grid == char)
            stats['tiles'][tile_type] = int(count)
        
        # Longitud del camino si existe
        if stats['valid']:
            path = self.get_path(map_grid)
            stats['path_length'] = len(path) if path else 0
        else:
            stats['path_length'] = -1
        
        return stats
    
    def print_statistics(self, map_grid: np.ndarray):
        """
        Imprime estadísticas del mapa de forma legible.
        
        Args:
            map_grid: Matriz del mapa
        """
        stats = self.get_statistics(map_grid)
        
        print("\n" + "=" * 40)
        print("  ESTADÍSTICAS DEL MAPA")
        print("=" * 40)
        print(f"Válido:            {'Sí' if stats['valid'] else 'No'}")
        print(f"Dimensiones:       {stats['size'][0]}x{stats['size'][1]}")
        print(f"Longitud camino:   {stats['path_length']}")
        print("\nDistribución de casillas:")
        
        for tile_type, count in stats['tiles'].items():
            char = self.config.get_char(tile_type)
            print(f"  {tile_type:10} ('{char}'): {count:3}")
        
        print("=" * 40)


# Ejemplo de uso
if __name__ == "__main__":
    from map_generator import MapGenerator
    
    # Cargar configuración
    config = ConfigManager('../config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración")
        exit(1)
    
    # Crear validador
    validator = MapValidator(config)
    
    print("\n--- Generando y validando mapas ---\n")
    
    # Generar y validar varios mapas
    valid_maps = 0
    total_maps = 5
    
    for i in range(total_maps):
        print(f"\n{'='*40}")
        print(f"  MAPA {i+1}/{total_maps}")
        print(f"{'='*40}")
        
        generator = MapGenerator(config, seed=i*123)
        map_grid = generator.generate_map(
            wall_density=0.15 + (i * 0.05),
            treasure_count=2,
            pit_count=2
        )
        
        generator.print_map()
        
        # Validar
        is_valid = validator.is_valid_map(map_grid, verbose=True)
        
        if is_valid:
            valid_maps += 1
            validator.print_statistics(map_grid)
            validator.visualize_path(map_grid)
    
    print(f"\n{'='*40}")
    print(f"Mapas válidos: {valid_maps}/{total_maps} ({valid_maps/total_maps*100:.1f}%)")
    print(f"{'='*40}")