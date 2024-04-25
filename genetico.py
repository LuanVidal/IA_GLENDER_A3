import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def main():
    # Leitura dos itens do arquivo
    items, capacity = read_items_from_file("./TRABALHO_PRATICO_1/instancias-mochila/KNAPDATA100.TXT")

    # Parâmetros do algoritmo genético
    population_size = 50
    crossover_rate = 0.8
    mutation_rate = 0.05  # Reduzindo a taxa de mutação
    num_generations = 1000  # Aumentando o número de gerações

    # Chamada para a função que implementa o algoritmo genético
    best_fitness_evolution, solution = genetic_algorithm_knapsack(items, capacity, population_size, crossover_rate, mutation_rate, num_generations)

    # Imprimir a solução encontrada
    print("Itens selecionados:")
    total_weight = 0
    total_value = 0
    for i, item in enumerate(items):
        if solution[i] == 1:
            total_weight += item.weight
            total_value += item.value
            print(f"Item {i+1}: Peso = {item.weight}, Valor = {item.value}")

    print(f"Peso total da mochila: {total_weight}")
    print(f"Valor total da mochila: {total_value}")

    # Plotar o gráfico da evolução do fitness
    plt.plot(best_fitness_evolution)
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Fitness')
    plt.show()

# Função para implementar o algoritmo genético para o Problema da Mochila
def genetic_algorithm_knapsack(items, capacity, population_size, crossover_rate, mutation_rate, num_generations):
    population = generate_initial_population(len(items), population_size)
    best_fitness_evolution = []

    with tqdm(total=num_generations) as pbar:
        for gen in range(num_generations):
            next_generation = []
            for _ in range(population_size):
                parent1 = tournament_selection(population, items, capacity)
                parent2 = roulette_selection(population, items, capacity)
                offspring = crossover(parent1, parent2, crossover_rate)
                mutate(offspring, mutation_rate)
                next_generation.extend(offspring)
            population = next_generation

            # Calcular o fitness da geração atual
            fitness_values = [fitness_function(individual, items, capacity) for individual in population]
            best_fitness = max(fitness_values)
            best_fitness_evolution.append(best_fitness)

            pbar.update(1)

    # Encontrar a melhor solução na última geração
    best_solution_index = fitness_values.index(best_fitness)
    best_solution = population[best_solution_index]
    return best_fitness_evolution, best_solution

# Função para gerar uma população inicial aleatória
def generate_initial_population(size, population_size):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(size)]  # 0 ou 1 (selecionado ou não selecionado)
        population.append(individual)
    return population

# Função de fitness para calcular o valor total da mochila
def fitness_function(solution, items, capacity):
    total_value = sum(item.value for item, selected in zip(items, solution) if selected == 1)
    total_weight = sum(item.weight for item, selected in zip(items, solution) if selected == 1)
    # Penalize soluções que excedam a capacidade da mochila
    if total_weight > capacity:
        total_value = 0
    return total_value

# Função para realizar a seleção por torneio
def tournament_selection(population, items, capacity):
    tournament_size = 5  # Tamanho do torneio
    tournament = random.sample(population, tournament_size)
    best_solution = max(tournament, key=lambda x: fitness_function(x, items, capacity))
    return best_solution

# Função para realizar a seleção por roleta
def roulette_selection(population, items, capacity):
    total_fitness = sum(fitness_function(individual, items, capacity) for individual in population)
    random_fitness = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for individual in population:
        cumulative_fitness += fitness_function(individual, items, capacity)
        if cumulative_fitness >= random_fitness:
            return individual

# Função para realizar o cruzamento (crossover)
def crossover(parent1, parent2, crossover_rate):
    # Completar a função de cruzamento
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    else:
        return parent1, parent2

# Função para realizar a mutação
def mutate(population, mutation_rate):
    for i in range(len(population)):
        solution = population[i]
        for j in range(len(solution)):
            if random.random() < mutation_rate:
                solution[j] = 1 if solution[j] == 0 else 0  # Alterna entre 0 e 1

# Função para ler os itens do arquivo
def read_items_from_file(file_path):
    items = []
    capacity = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        capacity = int(lines[0].strip())
        for line in lines[2:]:
            data = line.strip().split(',')
            weight = int(data[1])
            value = int(data[2])
            items.append(Item(weight, value))
    return items, capacity

if __name__ == "__main__":
    main()
