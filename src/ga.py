import random
import copy
from typing import List, Dict
from .model import Individual, CourseOffering
from .evaluation import evaluate
from .operators import interleaved_crossover, mutate_swap, repair_individual, sigma_scale


class GeneticSolver:
    def __init__(self, offers: List[CourseOffering], domains, cfg, n_cycles, n_rooms, n_teachers):
        self.offers = offers
        self.domains = domains
        self.cfg = cfg
        self.n_cycles = n_cycles
        self.n_rooms = n_rooms
        self.n_teachers = n_teachers
        self.subpopulations = self._segment_population()
        self.history: List[Dict] = []

    def _segment_population(self) -> Dict[str, Dict[int, List[int]]]:
        subpops = {"M": {}, "T": {}, "N": {}}
        for idx, off in enumerate(self.offers):
            turno = off.turno if off.turno in subpops else "M"
            horas = off.weekly_hours
            if horas not in subpops[turno]:
                subpops[turno][horas] = []
            subpops[turno][horas].append(idx)
        return subpops

    def selection_roulette(self, population: List[Individual]) -> Individual:
        total_fitness = sum(ind.scaled_fitness for ind in population)
        if total_fitness == 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in population:
            current += ind.scaled_fitness
            if current > pick:
                return ind
        return population[-1]

    def evolve(self, population: List[Individual], generations: int, mutation_rate: float):
        random.seed(self.cfg.seed)

        for ind in population:
            evaluate(ind, self.offers, self.n_cycles, self.n_rooms, self.n_teachers, self.cfg)
        sigma_scale(population, self.cfg)

        best_global = max(population, key=lambda x: x.fitness)
        best_penalty = best_global.penalty
        stagnation = 0

        for gen in range(generations):
            population.sort(key=lambda x: x.fitness, reverse=True)
            if population[0].penalty < best_penalty:
                best_penalty = population[0].penalty
                best_global = copy.deepcopy(population[0])
                stagnation = 0
            else:
                stagnation += 1

            avg_pen = sum(ind.penalty for ind in population) / len(population)
            self.history.append({"gen": gen, "best_penalty": best_penalty, "avg_penalty": avg_pen})

            if gen % 5 == 0 or gen == generations - 1:
                print(f"Gen {gen}: Mejor Penalty={best_penalty:.2f} Avg={avg_pen:.2f}")
            if best_penalty == 0 or stagnation >= self.cfg.max_stagnation:
                break

            new_pop: List[Individual] = []
            # Elitismo
            for i in range(min(self.cfg.elite_size, len(population))):
                new_pop.append(copy.deepcopy(population[i]))

            while len(new_pop) < self.cfg.population_size:
                p1 = self.selection_roulette(population)
                p2 = self.selection_roulette(population)
                child = interleaved_crossover(p1, p2)
                repair_individual(child, self.offers, self.domains, self.cfg)
                mutate_swap(child, mutation_rate)
                evaluate(child, self.offers, self.n_cycles, self.n_rooms, self.n_teachers, self.cfg)
                new_pop.append(child)

            sigma_scale(new_pop, self.cfg)
            population = new_pop

        return best_global
