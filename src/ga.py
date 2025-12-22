import random
import copy
from typing import List, Dict, Tuple
from .model import Individual, Gene, CourseOffering
from .evaluation import evaluate
from .initial_population import random_gene_for_offer

class GeneticSolver:
    def __init__(self, offers: List[CourseOffering], domains, cfg, n_cycles, n_rooms, n_teachers):
        self.offers = offers
        self.domains = domains
        self.cfg = cfg
        self.n_cycles = n_cycles
        self.n_rooms = n_rooms
        self.n_teachers = n_teachers
        
        # --- PASO 4: SEGMENTACIÓN EN SUBPOBLACIONES ---
        # Clasificamos los índices de los genes (cursos) en las 12 categorías
        # Estructura: self.subpopulations[turno][horas] = [indice_gen1, indice_gen2, ...]
        self.subpopulations = self._segment_population()

    def _segment_population(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Segmenta los cursos según el documento:
        Turnos: Mañana (M), Tarde (T), Noche (N)
        Horas: 6, 5, 4, 3 (u otros valores que aparezcan)
        """
        subpops = {
            "M": {}, "T": {}, "N": {}
        }
        
        for idx, off in enumerate(self.offers):
            # Obtener turno (asegurando que sea M, T o N)
            turno = off.turno if off.turno in subpops else "M"
            
            # Obtener carga horaria (agrupando si es necesario, el doc menciona 6,5,4,3)
            # Usamos weekly_hours como clave
            horas = off.weekly_hours
            
            if horas not in subpops[turno]:
                subpops[turno][horas] = []
            
            subpops[turno][horas].append(idx)
            
        return subpops

    def selection_roulette(self, population: List[Individual]) -> Individual:
        """Selección por Ruleta según documento."""
        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness == 0:
            return random.choice(population)
            
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind.fitness
            if current > pick:
                return ind
        return population[-1]

    def crossover_subpopulations(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        PASO 5: Operador de Cruce basado en Subpoblaciones.
        Itera por cada subpoblación (Turno/Horas) y decide de qué padre heredar 
        el bloque de genes correspondiente.
        """
        # Inicializamos genes vacíos
        child_genes = [None] * len(self.offers)
        
        # Iteramos por Turno (M, T, N)
        for turno, hours_dict in self.subpopulations.items():
            # Iteramos por Grupo Horario (6, 5, 4, 3...)
            for hours, gene_indices in hours_dict.items():
                
                # Para esta subpoblación completa, elegimos un padre fuente
                # El documento implica que se intercambian bloques.
                # Estrategia: 50% probabilidad de tomar TODOS los genes de esta subpoblación del Padre 1
                # y 50% del Padre 2. Esto preserva la consistencia dentro de la subpoblación.
                source_parent = parent1 if random.random() > 0.5 else parent2
                
                for idx in gene_indices:
                    # Copia profunda para evitar referencias cruzadas
                    original_gene = source_parent.genes[idx]
                    
                    # Si es laboratorio fijo (frozen), siempre se respeta (generalmente igual en ambos padres)
                    child_genes[idx] = copy.deepcopy(original_gene)

        # Por seguridad, si algún gen no estaba en subpoblaciones (raro), lo llenamos
        for i in range(len(child_genes)):
            if child_genes[i] is None:
                child_genes[i] = copy.deepcopy(parent1.genes[i])
                
        return Individual(genes=child_genes)

    def mutate(self, individual: Individual, mutation_rate: float):
        """Mutación por intercambio (regeneración)"""
        for i, gene in enumerate(individual.genes):
            if gene.frozen:
                continue
            
            if random.random() < mutation_rate:
                # Regenera el gen aleatoriamente dentro de su dominio válido
                individual.genes[i] = random_gene_for_offer(
                    self.offers[i], 
                    self.domains[i], 
                    self.cfg.N_SLOTS_PER_DAY
                )

    def evolve(self, population: List[Individual], generations: int, mutation_rate: float):
        # 1. Evaluación Inicial
        for ind in population:
            evaluate(ind, self.offers, self.n_cycles, self.n_rooms, self.n_teachers, self.cfg)
        
        # Elitismo: Guardar el mejor global
        best_global = max(population, key=lambda x: x.fitness)

        for gen in range(generations):
            # Ordenar población
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elitismo: Actualizar mejor global si encontramos uno mejor
            if population[0].fitness > best_global.fitness:
                best_global = copy.deepcopy(population[0])
            
            # Imprimir traza
            if gen % 10 == 0 or gen == generations - 1:
                print(f"Gen {gen}: Mejor Penalty = {best_global.penalty} (Fitness: {best_global.fitness:.5f})")
            
            if best_global.penalty == 0:
                print("¡Solución óptima (Penalty 0) encontrada!")
                break

            # Generar nueva población
            new_pop = [copy.deepcopy(best_global)] # El mejor pasa directo
            
            while len(new_pop) < len(population):
                # Selección
                p1 = self.selection_roulette(population)
                p2 = self.selection_roulette(population)
                
                # Cruce por Subpoblaciones
                child = self.crossover_subpopulations(p1, p2)
                
                # Mutación
                self.mutate(child, mutation_rate)
                
                # Evaluar hijo
                evaluate(child, self.offers, self.n_cycles, self.n_rooms, self.n_teachers, self.cfg)
                new_pop.append(child)
            
            population = new_pop

        return best_global