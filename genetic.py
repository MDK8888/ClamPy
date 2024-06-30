import time
import numpy as np

class GeneticAlgorithm:
    def __init__(self, num_param, size, min_param, max_param):
        self.population = np.random.uniform(min_param, max_param, (size, num_param))
        self.num_param = num_param
        self.size = size

    def _select_n_best(self, population, fitness, n, maximize, tournament_size=5):
        best = []
        for _ in range(n):
            tournament = np.random.choice(population, size=tournament_size, replace=False)
            scores = np.apply_along_axis(fitness, 1, tournament)
            if maximize:
                best_index = np.argmax(scores)
            else:
                best_index = np.argmin(scores)
            best.append(tournament[best_index])
        return np.array(best)

    def _crossover(self, population, p_cross):
        for i in range(self.size):
            if np.random.rand() < p_cross:
                parent1_index = np.random.randint(self.size)
                parent2_index = np.random.randint(self.size)
                crossover_point = np.random.randint(1, self.num_param)
                population[i, :crossover_point] = population[parent1_index, :crossover_point]
                population[i, crossover_point:] = population[parent2_index, crossover_point:]
        return population

    def _mutate(self, params, p_mutate, max_mutation):
        for i in range(self.num_param):
            if np.random.rand() < p_mutate:
                params[i] += np.random.uniform(-max_mutation, max_mutation)
                params[i] = np.clip(params[i], 0, 1)  # Ensure parameters stay within bounds
        return params

    def run(self, fitness, epochs=1000, p_cross=0.7, p_mutate=0.1, pct_best=0.2,
            max_mutation=0.1, maximize=True, verbose=False, history=False):
        start = time.time()
        n_best = max(int(self.size * pct_best), 1)
        history_list = []

        for i in range(1, epochs + 1):
            # Select the best individuals using tournament selection
            best = self._select_n_best(self.population, fitness, n_best, maximize)

            if verbose > 0 and i % verbose == 0:
                print("Epoch {:5d} | Fitness: {:10.3f} | Best Params: {}".format(
                    i, fitness(best[0]), best[0]))
                if history:
                    history_list.append(best[0])

            # Elitism: Preserve the best individuals
            self.population[:n_best] = best

            # Create new offspring through crossover and mutation
            for j in range(n_best, self.size):
                self.population[j] = self._mutate(self.population[j], p_mutate, max_mutation)
                if np.random.rand() < p_cross:
                    parent1_index = np.random.randint(n_best)
                    parent2_index = np.random.randint(n_best)
                    crossover_point = np.random.randint(1, self.num_param)
                    self.population[j, :crossover_point] = self.population[parent1_index, :crossover_point]
                    self.population[j, crossover_point:] = self.population[parent2_index, crossover_point:]

        self.best_params_history = np.stack(history_list, axis=0) if history else None

        if verbose > 0:
            print("Running time: {:.3f} seconds".format(time.time() - start))

        return self._select_n_best(self.population, fitness, 1, maximize)[0
