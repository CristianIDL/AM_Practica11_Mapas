"""
Módulo para cargar mapas de laberintos desde archivos .txt
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import os
import numpy as np
from typing import Optional, Tuple
from src.config_manager import ConfigManager


class MapLoader:
    """
    Carga y valida mapas de laberintos desde archivos de texto.
    Compatible con el sistema de configuración de caracteres.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Inicializa el cargador de mapas.
        
        Args:
            config: Instancia del ConfigManager con caracteres configurados
        """
        self.config = config
        self.current_map = None
        self.map_path = None
    
    def load_from_file(self, filepath: str, verbose: bool = True) -> Optional[np.ndarray]:
        """
        Carga un mapa desde un archivo .txt
        
        Args:
            filepath: Ruta al archivo .txt
            verbose: Si True, muestra información del proceso
            
        Returns:
            Matriz numpy con el mapa o None si hay error
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
            
            # Leer el archivo
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Limpiar líneas (eliminar saltos de línea y espacios al final)
            lines = [line.rstrip('\n\r') for line in lines if line.strip()]
            
            if len(lines) == 0:
                raise ValueError("El archivo está vacío")
            
            # Convertir a matriz
            map_grid = self._lines_to_array(lines)
            
            # Validar el mapa
            validation_errors = self._validate_map(map_grid)
            
            if validation_errors:
                if verbose:
                    print(f"Errores de validación en {filepath}:")
                    for error in validation_errors:
                        print(f"  - {error}")
                return None
            
            # Guardar información
            self.current_map = map_grid
            self.map_path = filepath
            
            if verbose:
                print(f"✓ Mapa cargado exitosamente desde: {filepath}")
                self._print_map_info(map_grid)
            
            return map_grid
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except ValueError as e:
            print(f"Error de formato: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado al cargar mapa: {e}")
            return None
    
    def _lines_to_array(self, lines: list) -> np.ndarray:
        """
        Convierte las líneas del archivo en una matriz numpy.
        
        Args:
            lines: Lista de strings (líneas del archivo)
            
        Returns:
            Matriz numpy de caracteres
        """
        # Obtener dimensiones
        num_rows = len(lines)
        num_cols = len(lines[0]) if lines else 0
        
        # Verificar que todas las filas tengan la misma longitud
        for i, line in enumerate(lines):
            if len(line) != num_cols:
                raise ValueError(
                    f"Fila {i} tiene longitud {len(line)}, se esperaba {num_cols}"
                )
        
        # Crear matriz
        map_grid = np.empty((num_rows, num_cols), dtype=str)
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                map_grid[i, j] = char
        
        return map_grid
    
    def _validate_map(self, map_grid: np.ndarray) -> list:
        """
        Valida que el mapa cumpla con los requisitos.
        
        Args:
            map_grid: Matriz del mapa
            
        Returns:
            Lista de errores encontrados (vacía si todo está bien)
        """
        errors = []
        
        # 1. Verificar que tiene START
        start_char = self.config.get_char('START')
        start_count = np.sum(map_grid == start_char)
        
        if start_count == 0:
            errors.append(f"No se encontró START ('{start_char}')")
        elif start_count > 1:
            errors.append(f"Se encontraron {start_count} START, debe haber solo 1")
        
        # 2. Verificar que tiene GOAL
        goal_char = self.config.get_char('GOAL')
        goal_count = np.sum(map_grid == goal_char)
        
        if goal_count == 0:
            errors.append(f"No se encontró GOAL ('{goal_char}')")
        elif goal_count > 1:
            errors.append(f"Se encontraron {goal_count} GOAL, debe haber solo 1")
        
        # 3. Verificar que todos los caracteres son válidos
        unique_chars = np.unique(map_grid)
        valid_chars = self.config.get_all_chars()
        
        invalid_chars = [char for char in unique_chars if char not in valid_chars]
        
        if invalid_chars:
            errors.append(
                f"Caracteres no válidos encontrados: {invalid_chars}"
            )
        
        # 4. Verificar dimensiones mínimas (al menos 3x3)
        if map_grid.shape[0] < 3 or map_grid.shape[1] < 3:
            errors.append(
                f"Dimensiones muy pequeñas: {map_grid.shape}. Mínimo 3x3"
            )
        
        return errors
    
    def get_map_dimensions(self) -> Optional[Tuple[int, int]]:
        """
        Obtiene las dimensiones del mapa actual.
        
        Returns:
            Tupla (filas, columnas) o None si no hay mapa cargado
        """
        if self.current_map is None:
            return None
        return self.current_map.shape
    
    def get_start_position(self) -> Optional[Tuple[int, int]]:
        """
        Encuentra la posición del START en el mapa actual.
        
        Returns:
            Tupla (fila, columna) o None si no se encuentra
        """
        if self.current_map is None:
            return None
        
        start_char = self.config.get_char('START')
        positions = np.where(self.current_map == start_char)
        
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None
    
    def get_goal_position(self) -> Optional[Tuple[int, int]]:
        """
        Encuentra la posición del GOAL en el mapa actual.
        
        Returns:
            Tupla (fila, columna) o None si no se encuentra
        """
        if self.current_map is None:
            return None
        
        goal_char = self.config.get_char('GOAL')
        positions = np.where(self.current_map == goal_char)
        
        if len(positions[0]) > 0:
            return (int(positions[0][0]), int(positions[1][0]))
        return None
    
    def get_treasure_positions(self) -> list:
        """
        Encuentra todas las posiciones de tesoros en el mapa actual.
        
        Returns:
            Lista de tuplas (fila, columna)
        """
        if self.current_map is None:
            return []
        
        treasure_char = self.config.get_char('TREASURE')
        positions = np.where(self.current_map == treasure_char)
        
        return list(zip(positions[0], positions[1]))
    
    def get_pit_positions(self) -> list:
        """
        Encuentra todas las posiciones de pozos en el mapa actual.
        
        Returns:
            Lista de tuplas (fila, columna)
        """
        if self.current_map is None:
            return []
        
        pit_char = self.config.get_char('PIT')
        positions = np.where(self.current_map == pit_char)
        
        return list(zip(positions[0], positions[1]))
    
    def print_map(self):
        """Imprime el mapa actual en consola."""
        if self.current_map is None:
            print("No hay mapa cargado.")
            return
        
        print("\n" + "=" * 50)
        print(f"  MAPA CARGADO")
        if self.map_path:
            print(f"  Archivo: {os.path.basename(self.map_path)}")
        print("=" * 50)
        
        for row in self.current_map:
            print(''.join(row))
        
        print("=" * 50)
    
    def _print_map_info(self, map_grid: np.ndarray):
        """
        Imprime información resumida del mapa.
        
        Args:
            map_grid: Matriz del mapa
        """
        print(f"  Dimensiones: {map_grid.shape[0]}x{map_grid.shape[1]}")
        
        # Contar elementos especiales
        treasure_char = self.config.get_char('TREASURE')
        pit_char = self.config.get_char('PIT')
        
        treasure_count = np.sum(map_grid == treasure_char)
        pit_count = np.sum(map_grid == pit_char)
        
        if treasure_count > 0:
            print(f"  Tesoros: {treasure_count}")
        if pit_count > 0:
            print(f"  Pozos: {pit_count}")
    
    def count_tiles(self) -> dict:
        """
        Cuenta todos los tipos de casillas en el mapa actual.
        
        Returns:
            Diccionario con el conteo de cada tipo
        """
        if self.current_map is None:
            return {}
        
        counts = {}
        
        for tile_type in ['START', 'GOAL', 'WALL', 'PATH', 'TREASURE', 'PIT']:
            char = self.config.get_char(tile_type)
            count = np.sum(self.current_map == char)
            counts[tile_type] = int(count)
        
        return counts
    
    def load_multiple_maps(self, directory: str, pattern: str = "*.txt") -> list:
        """
        Carga múltiples mapas desde un directorio.
        
        Args:
            directory: Ruta al directorio
            pattern: Patrón de archivos a cargar (default: "*.txt")
            
        Returns:
            Lista de tuplas (filepath, map_grid) para mapas válidos
        """
        import glob
        
        if not os.path.exists(directory):
            print(f"Error: El directorio {directory} no existe")
            return []
        
        # Buscar archivos que coincidan con el patrón
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        
        if not files:
            print(f"No se encontraron archivos .txt en {directory}")
            return []
        
        loaded_maps = []
        
        print(f"\n{'='*50}")
        print(f"Cargando mapas desde: {directory}")
        print(f"{'='*50}\n")
        
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Cargando {os.path.basename(filepath)}...")
            
            map_grid = self.load_from_file(filepath, verbose=False)
            
            if map_grid is not None:
                loaded_maps.append((filepath, map_grid))
                print(f"  ✓ Cargado correctamente")
            else:
                print(f"  ✗ Error al cargar")
            print()
        
        print(f"{'='*50}")
        print(f"Resumen: {len(loaded_maps)}/{len(files)} mapas cargados")
        print(f"{'='*50}\n")
        
        return loaded_maps


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    # Cargar configuración
    config = ConfigManager('config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración. Creando archivo por defecto...")
        config.create_default_config()
        config.load_config()
    
    # Crear cargador
    loader = MapLoader(config)
    
    print("\n" + "="*50)
    print("  PRUEBA DE MAP_LOADER")
    print("="*50)
    
    # Prueba 1: Cargar un mapa individual
    print("\n--- Prueba 1: Cargar mapa individual ---")
    test_map_path = "maps/test/ejemplo1.txt"
    
    if os.path.exists(test_map_path):
        map_grid = loader.load_from_file(test_map_path)
        
        if map_grid is not None:
            loader.print_map()
            
            print("\nInformación adicional:")
            print(f"  Start: {loader.get_start_position()}")
            print(f"  Goal:  {loader.get_goal_position()}")
            print(f"  Tesoros: {loader.get_treasure_positions()}")
            print(f"  Pozos: {loader.get_pit_positions()}")
            
            print("\nConteo de casillas:")
            counts = loader.count_tiles()
            for tile_type, count in counts.items():
                print(f"  {tile_type:10}: {count}")
    else:
        print(f"Archivo de prueba no encontrado: {test_map_path}")
        print("Creando mapa de ejemplo...")
        
        # Crear directorio si no existe
        os.makedirs("maps/test", exist_ok=True)
        
        # Crear mapa de ejemplo simple (formato estándar de la práctica)
        example_map = [
            "##########G#",
            "#''''''''''#",
            "#''T'''''''#",
            "#''''''''''#",
            "#'''X''''''#",
            "#''''''''''#",
            "#''''''''''#",
            "#''''''''''#",
            "#''''''''''#",
            "#''''''''''#",
            "#''''''''''#",
            "#S##########"
        ]
        
        with open(test_map_path, 'w', encoding='utf-8') as f:
            for line in example_map:
                f.write(line + '\n')
        
        print(f"Mapa de ejemplo creado en: {test_map_path}")
        print("Ejecuta el script nuevamente para cargarlo.")
    
    # Prueba 2: Cargar múltiples mapas (si existen)
    print("\n\n--- Prueba 2: Cargar múltiples mapas ---")
    maps_directory = "maps/test"
    
    if os.path.exists(maps_directory):
        loaded_maps = loader.load_multiple_maps(maps_directory)
        
        if loaded_maps:
            print(f"\nSe cargaron {len(loaded_maps)} mapas:")
            for filepath, _ in loaded_maps:
                print(f"  - {os.path.basename(filepath)}")
    else:
        print(f"Directorio no encontrado: {maps_directory}")