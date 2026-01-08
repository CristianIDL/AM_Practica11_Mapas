"""
Módulo para generar mapas validados automáticamente.
Garantiza que todos los mapas generados tengan solución.
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import numpy as np
import os
from typing import Optional, Tuple, List
from src.config_manager import ConfigManager
from src.map_generator import MapGenerator
from src.map_validator import MapValidator


class ValidatedMapGenerator:
    """
    Genera mapas de laberintos asegurando que sean válidos (tengan solución).
    """
    
    def __init__(self, config: ConfigManager, max_attempts: int = 50):
        """
        Inicializa el generador con validación.
        
        Args:
            config: Instancia del ConfigManager
            max_attempts: Número máximo de intentos para generar un mapa válido
        """
        self.config = config
        self.max_attempts = max_attempts
        self.validator = MapValidator(config)
        self.last_attempts = 0
    
    def generate_valid_map(self,
                          wall_density: float = 0.15,
                          treasure_count: int = 3,
                          pit_count: int = 2,
                          seed: Optional[int] = None,
                          verbose: bool = True) -> Tuple[np.ndarray, int]:
        """
        Genera un mapa válido (con camino de S a G).
        
        Args:
            wall_density: Densidad de paredes (0.0 a 1.0)
            treasure_count: Número de tesoros
            pit_count: Número de pozos
            seed: Semilla inicial (se incrementará si falla)
            verbose: Si True, muestra intentos
            
        Returns:
            Tupla (map_grid, seed_used)
            
        Raises:
            RuntimeError: Si no se puede generar un mapa válido después de max_attempts
        """
        base_seed = seed if seed is not None else np.random.randint(0, 999999)
        
        if verbose:
            print(f"\nGenerando mapa válido (densidad paredes: {wall_density:.2f})...")
        
        for attempt in range(self.max_attempts):
            current_seed = base_seed + attempt
            
            # Generar mapa
            generator = MapGenerator(self.config, seed=current_seed)
            map_grid = generator.generate_map(
                wall_density=wall_density,
                treasure_count=treasure_count,
                pit_count=pit_count
            )
            
            # Validar
            is_valid = self.validator.is_valid_map(map_grid, verbose=False)
            
            if is_valid:
                self.last_attempts = attempt + 1
                
                if verbose:
                    if attempt > 0:
                        print(f"  ✓ Mapa válido encontrado después de {attempt + 1} intentos")
                    else:
                        print(f"  ✓ Mapa válido generado en el primer intento")
                    print(f"  Semilla utilizada: {current_seed}")
                
                return map_grid, current_seed
            
            elif verbose and (attempt + 1) % 10 == 0:
                print(f"  Intento {attempt + 1}/{self.max_attempts}...")
        
        # No se pudo generar mapa válido
        raise RuntimeError(
            f"No se pudo generar un mapa válido después de {self.max_attempts} intentos. "
            f"Intente reducir wall_density o aumentar max_attempts."
        )
    
    def generate_batch(self,
                      count: int,
                      wall_density: float = 0.15,
                      treasure_count: int = 3,
                      pit_count: int = 2,
                      output_dir: Optional[str] = None,
                      verbose: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Genera múltiples mapas válidos.
        
        Args:
            count: Número de mapas a generar
            wall_density: Densidad de paredes
            treasure_count: Número de tesoros
            pit_count: Número de pozos
            output_dir: Si se proporciona, guarda los mapas en este directorio
            verbose: Si True, muestra progreso
            
        Returns:
            Lista de tuplas (map_grid, seed)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  GENERACIÓN DE LOTE: {count} mapas")
            print(f"{'='*60}")
        
        maps = []
        total_attempts = 0
        
        for i in range(count):
            if verbose:
                print(f"\n[{i+1}/{count}] Generando mapa...")
            
            try:
                map_grid, seed = self.generate_valid_map(
                    wall_density=wall_density,
                    treasure_count=treasure_count,
                    pit_count=pit_count,
                    seed=None,  # Cada mapa con semilla diferente
                    verbose=verbose
                )
                
                maps.append((map_grid, seed))
                total_attempts += self.last_attempts
                
                # Guardar si se especificó directorio
                if output_dir:
                    self._save_map(map_grid, seed, output_dir, i+1)
                
            except RuntimeError as e:
                print(f"  ✗ Error: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  RESUMEN")
            print(f"{'='*60}")
            print(f"  Mapas generados:     {len(maps)}/{count}")
            print(f"  Intentos totales:    {total_attempts}")
            print(f"  Promedio por mapa:   {total_attempts/len(maps):.1f}" if maps else "  N/A")
            
            if output_dir:
                print(f"  Guardados en:        {output_dir}")
            
            print(f"{'='*60}")
        
        return maps
    
    def _save_map(self, map_grid: np.ndarray, seed: int, directory: str, index: int):
        """
        Guarda un mapa en un archivo.
        
        Args:
            map_grid: Matriz del mapa
            seed: Semilla utilizada
            directory: Directorio de salida
            index: Índice del mapa
        """
        os.makedirs(directory, exist_ok=True)
        
        filename = f"map_{index:03d}_seed{seed}.txt"
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for row in map_grid:
                f.write(''.join(row) + '\n')
    
    def generate_training_set(self,
                             lab_type: str,
                             count: int = 10,
                             output_dir: Optional[str] = None) -> List[Tuple[np.ndarray, int]]:
        """
        Genera un conjunto de mapas para entrenamiento con configuración según el tipo.
        
        Args:
            lab_type: "lab1" o "lab2"
            count: Número de mapas
            output_dir: Directorio para guardar mapas
            
        Returns:
            Lista de tuplas (map_grid, seed)
        """
        lab_type = lab_type.lower()
        
        if lab_type == "lab1":
            # Lab1: Solo paredes, sin tesoros ni pozos
            print(f"\nGenerando conjunto Lab1 (simple)...")
            return self.generate_batch(
                count=count,
                wall_density=0.15,
                treasure_count=0,
                pit_count=0,
                output_dir=output_dir
            )
        
        elif lab_type == "lab2":
            # Lab2: Con tesoros y pozos
            print(f"\nGenerando conjunto Lab2 (tesoros)...")
            return self.generate_batch(
                count=count,
                wall_density=0.12,  # Menor densidad para dar espacio
                treasure_count=3,
                pit_count=2,
                output_dir=output_dir
            )
        
        else:
            raise ValueError(f"Tipo de laberinto no válido: {lab_type}")
    
    def analyze_generation_difficulty(self,
                                     density_range: Tuple[float, float] = (0.05, 0.30),
                                     steps: int = 6,
                                     trials_per_density: int = 10) -> dict:
        """
        Analiza qué densidades de paredes son más propensas a generar mapas inválidos.
        
        Args:
            density_range: Rango de densidades a probar (min, max)
            steps: Número de densidades a probar
            trials_per_density: Intentos por cada densidad
            
        Returns:
            Diccionario con estadísticas
        """
        print(f"\n{'='*60}")
        print(f"  ANÁLISIS DE DIFICULTAD DE GENERACIÓN")
        print(f"{'='*60}")
        
        densities = np.linspace(density_range[0], density_range[1], steps)
        results = {}
        
        for density in densities:
            attempts_list = []
            
            print(f"\nDensidad {density:.2f}:")
            
            for trial in range(trials_per_density):
                try:
                    _, _ = self.generate_valid_map(
                        wall_density=density,
                        treasure_count=3,
                        pit_count=2,
                        verbose=False
                    )
                    attempts_list.append(self.last_attempts)
                
                except RuntimeError:
                    attempts_list.append(self.max_attempts)
            
            avg_attempts = np.mean(attempts_list)
            success_rate = sum(1 for a in attempts_list if a < self.max_attempts) / trials_per_density * 100
            
            results[density] = {
                'avg_attempts': avg_attempts,
                'success_rate': success_rate,
                'attempts_list': attempts_list
            }
            
            print(f"  Intentos promedio: {avg_attempts:.1f}")
            print(f"  Tasa de éxito:     {success_rate:.0f}%")
        
        # Recomendación
        print(f"\n{'='*60}")
        print(f"  RECOMENDACIONES")
        print(f"{'='*60}")
        
        best_density = min(results.keys(), key=lambda d: results[d]['avg_attempts'])
        print(f"  Densidad más eficiente: {best_density:.2f}")
        print(f"  (Promedio {results[best_density]['avg_attempts']:.1f} intentos)")
        
        safe_densities = [d for d, r in results.items() if r['success_rate'] == 100]
        if safe_densities:
            print(f"\n  Densidades con 100% éxito:")
            for d in safe_densities:
                print(f"    - {d:.2f}")
        
        print(f"{'='*60}")
        
        return results


# Ejemplo de uso
if __name__ == "__main__":
    from src.map_validator import MapValidator
    
    # Cargar configuración
    config = ConfigManager('config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración")
        exit(1)
    
    # Crear generador validado
    generator = ValidatedMapGenerator(config, max_attempts=50)
    validator = MapValidator(config)
    
    print("\n" + "="*70)
    print("  PRUEBA DE VALIDATED MAP GENERATOR")
    print("="*70)
    
    # Prueba 1: Generar un solo mapa válido
    print("\n--- Prueba 1: Generar mapa único ---")
    try:
        map_grid, seed = generator.generate_valid_map(
            wall_density=0.20,
            treasure_count=3,
            pit_count=2,
            seed=42,
            verbose=True
        )
        
        print("\nMapa generado:")
        for row in map_grid:
            print(''.join(row))
        
        # Verificar con validator
        validator.print_statistics(map_grid)
        validator.visualize_path(map_grid)
        
    except RuntimeError as e:
        print(f"Error: {e}")
    
    # Prueba 2: Generar lote
    print("\n\n--- Prueba 2: Generar lote de mapas ---")
    maps = generator.generate_batch(
        count=5,
        wall_density=0.15,
        treasure_count=3,
        pit_count=2,
        output_dir="maps/generated/batch_test",
        verbose=True
    )
    
    print(f"\n{len(maps)} mapas generados exitosamente")
    
    # Prueba 3: Generar conjunto de entrenamiento
    print("\n\n--- Prueba 3: Generar conjunto Lab2 ---")
    training_maps = generator.generate_training_set(
        lab_type="lab2",
        count=3,
        output_dir="maps/training/lab2"
    )
    
    # Prueba 4: Análisis de dificultad (comentado por tiempo)
    """
    print("\n\n--- Prueba 4: Análisis de dificultad ---")
    analysis = generator.analyze_generation_difficulty(
        density_range=(0.10, 0.25),
        steps=4,
        trials_per_density=5
    )
    """
    
    print("\n" + "="*70)
    print("  Pruebas completadas exitosamente")
    print("="*70)