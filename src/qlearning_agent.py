"""
Módulo para el agente de Q-Learning.
Implementa el algoritmo de aprendizaje por refuerzo para resolver laberintos.
Autor: CristianIDL
Fecha: Diciembre 2025
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Dict
from src.config_manager import ConfigManager
from src.reward_system import RewardSystemBase, RewardSystemLab1, RewardSystemLab2


class QLearningAgent:
    """
    Agente que aprende a navegar laberintos usando Q-Learning.
    Compatible con Lab1 (simple) y Lab2 (tesoros).
    """
    
    # Acciones posibles: 0=arriba, 1=abajo, 2=izquierda, 3=derecha
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = ['↑ Arriba', '↓ Abajo', '← Izquierda', '→ Derecha']
    
    def __init__(self, 
                 config: ConfigManager,
                 reward_system: RewardSystemBase,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            config: Instancia del ConfigManager
            reward_system: Sistema de recompensas (Lab1 o Lab2)
            alpha: Tasa de aprendizaje (learning rate)
            gamma: Factor de descuento (discount factor)
            epsilon: Probabilidad inicial de exploración
            epsilon_min: Probabilidad mínima de exploración
            epsilon_decay: Factor de decaimiento de epsilon
        """
        self.config = config
        self.reward_system = reward_system
        
        # Hiperparámetros
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q (se inicializa al entrenar)
        self.Q = None
        self.map_shape = None
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'successful_episodes': 0,
            'episode_rewards': [],
            'episode_lengths': []
        }
    
    def _state_to_index(self, row: int, col: int) -> int:
        """
        Convierte posición (fila, columna) a índice de estado.
        
        Args:
            row: Fila
            col: Columna
            
        Returns:
            Índice del estado
        """
        return row * self.map_shape[1] + col
    
    def _index_to_state(self, index: int) -> Tuple[int, int]:
        """
        Convierte índice de estado a posición (fila, columna).
        
        Args:
            index: Índice del estado
            
        Returns:
            Tupla (fila, columna)
        """
        row = index // self.map_shape[1]
        col = index % self.map_shape[1]
        return (row, col)
    
    def _get_next_position(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Calcula la siguiente posición dada una acción.
        
        Args:
            position: Posición actual (fila, columna)
            action: Acción a tomar
            
        Returns:
            Nueva posición
        """
        row, col = position
        
        if action == 0:    # Arriba
            return (row - 1, col)
        elif action == 1:  # Abajo
            return (row + 1, col)
        elif action == 2:  # Izquierda
            return (row, col - 1)
        elif action == 3:  # Derecha
            return (row, col + 1)
        
        return position
    
    def _is_out_of_bounds(self, position: Tuple[int, int]) -> bool:
        """
        Verifica si una posición está fuera de los límites del mapa.
        
        Args:
            position: Posición a verificar
            
        Returns:
            True si está fuera de límites
        """
        row, col = position
        return not (0 <= row < self.map_shape[0] and 0 <= col < self.map_shape[1])
    
    def _choose_action(self, state_index: int, greedy: bool = False) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state_index: Índice del estado actual
            greedy: Si True, siempre elige la mejor acción (sin exploración)
            
        Returns:
            Acción seleccionada
        """
        if not greedy and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.choice(self.ACTIONS)
        else:
            # Explotación: mejor acción según Q
            return np.argmax(self.Q[state_index])
    
    def train(self, 
              map_grid: np.ndarray,
              episodes: int = 5000,
              max_steps_per_episode: int = 300,
              verbose: bool = True,
              verbose_interval: int = 500) -> Dict:
        """
        Entrena el agente en un mapa específico.
        
        Args:
            map_grid: Matriz del mapa
            episodes: Número de episodios de entrenamiento
            max_steps_per_episode: Máximo de pasos por episodio
            verbose: Si True, muestra progreso
            verbose_interval: Cada cuántos episodios mostrar información
            
        Returns:
            Diccionario con estadísticas de entrenamiento
        """
        # Inicializar tabla Q si es necesaria
        self.map_shape = map_grid.shape
        num_states = self.map_shape[0] * self.map_shape[1]
        
        if self.Q is None:
            self.Q = np.zeros((num_states, len(self.ACTIONS)))
            if verbose:
                print(f"Tabla Q inicializada: {num_states} estados x {len(self.ACTIONS)} acciones")
        
        # Encontrar posición inicial
        start_char = self.config.get_char('START')
        start_positions = np.where(map_grid == start_char)
        start_pos = (int(start_positions[0][0]), int(start_positions[1][0]))
        
        goal_char = self.config.get_char('GOAL')
        wall_char = self.config.get_char('WALL')
        treasure_char = self.config.get_char('TREASURE')
        path_char = self.config.get_char('PATH')
        
        if verbose:
            print(f"\nIniciando entrenamiento...")
            print(f"  Episodios: {episodes}")
            print(f"  Posición inicial: {start_pos}")
            print(f"  Epsilon inicial: {self.epsilon:.3f}")
            print(f"  Epsilon mínimo: {self.epsilon_min:.3f}")
        
        # Entrenamiento
        for episode in range(episodes):
            # Reiniciar episodio con copia del mapa (para consumir tesoros)
            episode_map = map_grid.copy()
            current_pos = start_pos
            self.reward_system.reset_episode()
            
            episode_reward = 0
            steps = 0
            reached_goal = False
            
            # Ejecutar episodio
            visited_positions = {}  # Para detectar loops
            
            for step in range(max_steps_per_episode):
                steps += 1
                
                # Detectar loop (visitó la misma posición muchas veces)
                if current_pos in visited_positions:
                    visited_positions[current_pos] += 1
                    # Si visita la misma posición >10 veces, penalizar y terminar
                    if visited_positions[current_pos] > 10:
                        episode_reward -= 50  # Penalización por loop
                        break
                else:
                    visited_positions[current_pos] = 1
                
                # Elegir acción
                state_idx = self._state_to_index(*current_pos)
                action = self._choose_action(state_idx)
                
                # Calcular siguiente posición
                next_pos = self._get_next_position(current_pos, action)
                
                # Verificar límites
                out_of_bounds = self._is_out_of_bounds(next_pos)
                
                if out_of_bounds:
                    # Fuera de límites: penalización y no se mueve
                    reward, done = self.reward_system.get_step_reward(
                        episode_map, current_pos, action, current_pos, out_of_bounds=True
                    )
                    next_pos = current_pos
                else:
                    # Dentro de límites
                    cell = episode_map[next_pos]
                    
                    if cell == wall_char:
                        # Chocó con pared: penalización y no se mueve
                        reward, done = self.reward_system.get_step_reward(
                            episode_map, current_pos, action, next_pos, out_of_bounds=False
                        )
                        next_pos = current_pos
                    else:
                        # Movimiento válido
                        reward, done = self.reward_system.get_step_reward(
                            episode_map, current_pos, action, next_pos, out_of_bounds=False
                        )
                        
                        # NUEVO: Consumir tesoro si fue recogido
                        if cell == treasure_char:
                            episode_map[next_pos] = path_char
                        
                        if cell == goal_char:
                            reached_goal = True
                
                # Actualizar Q-Learning
                next_state_idx = self._state_to_index(*next_pos)
                
                if done:
                    # Estado terminal
                    target = reward
                else:
                    # Q-Learning update
                    target = reward + self.gamma * np.max(self.Q[next_state_idx])
                
                self.Q[state_idx, action] += self.alpha * (target - self.Q[state_idx, action])
                
                # Actualizar estado
                episode_reward += reward
                current_pos = next_pos
                
                if done:
                    break
            
            # Aplicar bonificación final (solo Lab2)
            if reached_goal and isinstance(self.reward_system, RewardSystemLab2):
                final_bonus = self.reward_system.get_final_reward(reached_goal=True)
                episode_reward += final_bonus
            
            # Actualizar epsilon (decaimiento)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Registrar estadísticas
            self.training_stats['episodes'] += 1
            self.training_stats['total_steps'] += steps
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(steps)
            
            if reached_goal:
                self.training_stats['successful_episodes'] += 1
            
            # Mostrar progreso
            if verbose and (episode + 1) % verbose_interval == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-verbose_interval:])
                avg_steps = np.mean(self.training_stats['episode_lengths'][-verbose_interval:])
                success_rate = self.training_stats['successful_episodes'] / (episode + 1) * 100
                
                print(f"Episodio {episode + 1}/{episodes} | "
                      f"ε={self.epsilon:.3f} | "
                      f"Reward promedio={avg_reward:+.1f} | "
                      f"Pasos promedio={avg_steps:.1f} | "
                      f"Éxito={success_rate:.1f}%")
        
        if verbose:
            print(f"\n¡Entrenamiento completado!")
            self._print_training_summary()
        
        return self.training_stats
    
    def execute(self, 
                map_grid: np.ndarray,
                max_steps: int = 200,
                verbose: bool = True) -> Dict:
        """
        Ejecuta el agente entrenado en un mapa (sin aprendizaje).
        
        Args:
            map_grid: Matriz del mapa
            max_steps: Número máximo de pasos
            verbose: Si True, muestra información
            
        Returns:
            Diccionario con resultados de la ejecución
        """
        if self.Q is None:
            raise ValueError("El agente no ha sido entrenado. Ejecute train() primero.")
        
        # Crear copia del mapa para consumir tesoros
        episode_map = map_grid.copy()
        
        # Encontrar posición inicial
        start_char = self.config.get_char('START')
        start_positions = np.where(episode_map == start_char)
        start_pos = (int(start_positions[0][0]), int(start_positions[1][0]))
        
        goal_char = self.config.get_char('GOAL')
        wall_char = self.config.get_char('WALL')
        treasure_char = self.config.get_char('TREASURE')
        path_char = self.config.get_char('PATH')
        
        # Reiniciar sistema de recompensas
        self.reward_system.reset_episode()
        
        # Variables de ejecución
        current_pos = start_pos
        path = [start_pos]
        actions_taken = []
        total_reward = 0
        reached_goal = False
        
        # Ejecutar
        for step in range(max_steps):
            # Elegir mejor acción (greedy)
            state_idx = self._state_to_index(*current_pos)
            action = self._choose_action(state_idx, greedy=True)
            
            # Calcular siguiente posición
            next_pos = self._get_next_position(current_pos, action)
            
            # Verificar límites
            if self._is_out_of_bounds(next_pos):
                if verbose:
                    print(f"  Paso {step + 1}: Intentó salir del mapa")
                break
            
            # Verificar pared
            if episode_map[next_pos] == wall_char:
                if verbose:
                    print(f"  Paso {step + 1}: Chocó con pared")
                break
            
            # Obtener recompensa
            reward, done = self.reward_system.get_step_reward(
                episode_map, current_pos, action, next_pos
            )
            
            # Consumir tesoro si fue recogido
            if episode_map[next_pos] == treasure_char:
                episode_map[next_pos] = path_char
            
            # Actualizar
            total_reward += reward
            current_pos = next_pos
            path.append(current_pos)
            actions_taken.append(action)
            
            if episode_map[current_pos] == goal_char:
                reached_goal = True
                if verbose:
                    print(f"  ¡Meta alcanzada en {step + 1} pasos!")
                break
        
        # Bonificación final (Lab2)
        if reached_goal and isinstance(self.reward_system, RewardSystemLab2):
            final_bonus = self.reward_system.get_final_reward(reached_goal=True)
            total_reward += final_bonus
        
        # Resultados
        results = {
            'path': path,
            'actions': actions_taken,
            'steps': len(path) - 1,
            'total_reward': total_reward,
            'reached_goal': reached_goal
        }
        
        # Estadísticas adicionales para Lab2
        if isinstance(self.reward_system, RewardSystemLab2):
            episode_stats = self.reward_system.get_episode_score(reached_goal)
            results.update(episode_stats)
        
        if verbose:
            self._print_execution_summary(results)
        
        return results
    
    def visualize_path(self, map_grid: np.ndarray, path: List[Tuple[int, int]]):
        """
        Visualiza el camino recorrido en el mapa.
        
        Args:
            map_grid: Matriz del mapa
            path: Lista de posiciones del camino
        """
        # Crear copia para visualización
        visual_map = map_grid.copy()
        
        start_char = self.config.get_char('START')
        goal_char = self.config.get_char('GOAL')
        
        # Marcar camino (excepto START y GOAL)
        for pos in path:
            if visual_map[pos] not in [start_char, goal_char]:
                visual_map[pos] = '·'
        
        print("\n" + "="*50)
        print("  VISUALIZACIÓN DEL CAMINO")
        print("="*50)
        
        for row in visual_map:
            print(''.join(row))
        
        print("="*50)
        print(f"Longitud del camino: {len(path)} posiciones")
    
    def reset_epsilon(self):
        """Reinicia epsilon a su valor inicial."""
        self.epsilon = self.epsilon_initial
    
    def _print_training_summary(self):
        """Imprime resumen de entrenamiento."""
        stats = self.training_stats
        
        print("\n" + "="*60)
        print("  RESUMEN DE ENTRENAMIENTO")
        print("="*60)
        print(f"  Episodios totales:       {stats['episodes']}")
        print(f"  Pasos totales:           {stats['total_steps']}")
        print(f"  Episodios exitosos:      {stats['successful_episodes']}")
        print(f"  Tasa de éxito:           {stats['successful_episodes']/stats['episodes']*100:.2f}%")
        
        if stats['episode_rewards']:
            print(f"\n  Recompensa promedio:     {np.mean(stats['episode_rewards']):.2f}")
            print(f"  Recompensa máxima:       {np.max(stats['episode_rewards']):.2f}")
            print(f"  Recompensa mínima:       {np.min(stats['episode_rewards']):.2f}")
        
        if stats['episode_lengths']:
            print(f"\n  Pasos promedio:          {np.mean(stats['episode_lengths']):.2f}")
            print(f"  Pasos mínimos:           {np.min(stats['episode_lengths'])}")
            print(f"  Pasos máximos:           {np.max(stats['episode_lengths'])}")
        
        print(f"\n  Epsilon final:           {self.epsilon:.4f}")
        print("="*60)
    
    def _print_execution_summary(self, results: Dict):
        """Imprime resumen de ejecución."""
        print("\n" + "="*60)
        print("  RESUMEN DE EJECUCIÓN")
        print("="*60)
        print(f"  Pasos dados:             {results['steps']}")
        print(f"  Recompensa total:        {results['total_reward']:+.2f}")
        print(f"  Meta alcanzada:          {'Sí' if results['reached_goal'] else 'No'}")
        
        # Info adicional para Lab2
        if 'treasures_collected' in results:
            print(f"\n  Tesoros recolectados:    {results['treasures_collected']}/{results['total_treasures']}")
        
        if 'time_bonus' in results:
            print(f"  Bonificación por tiempo: +{results['time_bonus']}")
        
        print("="*60)


# Ejemplo de uso
if __name__ == "__main__":
    from src.validated_map_generator import ValidatedMapGenerator
    from src.map_validator import MapValidator
    from src.reward_system import create_reward_system
    
    # Configuración
    config = ConfigManager('config/casillas.txt')
    
    if not config.load_config():
        print("Error al cargar configuración")
        exit(1)
    
    print("\n" + "="*70)
    print("  PRUEBA DE Q-LEARNING AGENT")
    print("="*70)
    
    # Generar mapa válido para Lab2
    print("\n--- Generando mapa Lab2 (con validación) ---")
    generator = ValidatedMapGenerator(config, max_attempts=50)
    
    try:
        map_grid, seed = generator.generate_valid_map(
            wall_density=0.15,
            treasure_count=3,
            pit_count=2,
            seed=42,
            verbose=True
        )
        
        # Mostrar mapa
        print("\nMapa generado:")
        for row in map_grid:
            print(''.join(row))
        
    except RuntimeError as e:
        print(f"ERROR: {e}")
        exit(1)
    
    # Verificar con validator
    validator = MapValidator(config)
    validator.is_valid_map(map_grid, verbose=True)
    
    # Crear sistema de recompensas
    reward_system = create_reward_system("lab2", config)
    reward_system.print_config()
    
    # Crear y entrenar agente
    print("\n--- Entrenando agente ---")
    agent = QLearningAgent(
        config=config,
        reward_system=reward_system,
        alpha=0.2,            # Mayor learning rate
        gamma=0.98,           # Muy alto para valorar meta lejana
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995  # Decay más lento
    )
    
    agent.train(
        map_grid=map_grid,
        episodes=5000,        # Más episodios
        max_steps_per_episode=200,  # REDUCIDO para forzar eficiencia
        verbose=True,
        verbose_interval=1000
    )
    
    # Ejecutar agente entrenado
    print("\n--- Ejecutando agente entrenado ---")
    results = agent.execute(map_grid, max_steps=300, verbose=True)
    
    # Visualizar camino
    agent.visualize_path(map_grid, results['path'])
    
    print("\n" + "="*70)
    print("  Prueba completada exitosamente")
    print("="*70)