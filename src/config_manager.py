"""
Módulo para gestionar la configuración de caracteres del laberinto.
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import os
from typing import Dict, List, Optional

class ConfigManager:
    """
    Gestiona la lectura y validación del archivo de configuración
    que define los caracteres para cada tipo de casilla.
    """
    
    # Tipos de casillas
    REQUIRED_TYPES = ['START', 'GOAL', 'WALL', 'PATH', 'TREASURE', 'PIT']
    
    def __init__(self, config_path: str = 'config/casillas.txt'):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.char_mapping: Dict[str, str] = {}
        self.reverse_mapping: Dict[str, str] = {}
        
    def load_config(self) -> bool:
        """
        Carga el archivo de configuración y crea los mapeos.
        
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"No se encontró el archivo: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            # Validar que tenemos exactamente 6 caracteres
            if len(lines) != 6:
                raise ValueError(f"Se esperaban 6 caracteres, se encontraron {len(lines)}")
            
            # Crear mapeos
            self.char_mapping = {
                'START': lines[0],
                'GOAL': lines[1],
                'WALL': lines[2],
                'PATH': lines[3],
                'TREASURE': lines[4],
                'PIT': lines[5]
            }
            
            # Crear mapeo inverso (caracter -> tipo)
            self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}
            
            # Validar que no hay caracteres duplicados
            if len(self.reverse_mapping) != 6:
                raise ValueError("Hay caracteres duplicados en la configuración")
            
            print("Configuración cargada exitosamente")
            self._print_mapping()
            return True
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
        except ValueError as e:
            print(f"Error de validación: {e}")
            return False
        except Exception as e:
            print(f"Error inesperado: {e}")
            return False
    
    def get_char(self, tile_type: str) -> Optional[str]:
        """
        Obtiene el carácter asociado a un tipo de casilla.
        
        Args:
            tile_type: Tipo de casilla (START, GOAL, WALL, etc.)
            
        Returns:
            Carácter correspondiente o None si no existe
        """
        return self.char_mapping.get(tile_type.upper())
    
    def get_type(self, char: str) -> Optional[str]:
        """
        Obtiene el tipo de casilla asociado a un carácter.
        
        Args:
            char: Carácter a buscar
            
        Returns:
            Tipo de casilla o None si no existe
        """
        return self.reverse_mapping.get(char)
    
    def validate_char(self, char: str) -> bool:
        """
        Valida si un carácter está en la configuración.
        
        Args:
            char: Carácter a validar
            
        Returns:
            True si el carácter es válido, False en caso contrario
        """
        return char in self.reverse_mapping
    
    def _print_mapping(self):
        """Imprime el mapeo de caracteres de forma legible."""
        print("\nMapeo de caracteres:")
        print("-" * 30)
        for tile_type, char in self.char_mapping.items():
            print(f"  {tile_type:12} : '{char}'")
        print("-" * 30)
    
    def create_default_config(self):
        """
        Crea un archivo de configuración por defecto si no existe.
        """
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        default_chars = ['S', 'G', '#', "'", 'T', 'X']
        
        with open(self.config_path, 'w', encoding='utf-8') as file:
            for char in default_chars:
                file.write(f"{char}\n")
        
        print(f"Archivo de configuración creado en: {self.config_path}")
    
    def get_all_chars(self) -> List[str]:
        """
        Obtiene todos los caracteres configurados.
        
        Returns:
            Lista con todos los caracteres
        """
        return list(self.char_mapping.values())