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
    population_size = 150
    crossover_rate = 0.7
    mutation_rate = 0.05  # Reduzindo a taxa de mutação
    num_generations = 300  # Aumentando o número de gerações
    tournament_size = 5  # Tamanho do torneio

    # Chamada para a função que implementa o algoritmo genético
    best_fitness_evolution, solution = genetic_algorithm_knapsack(items, capacity, population_size, crossover_rate, mutation_rate, num_generations, tournament_size)

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
def genetic_algorithm_knapsack(items, capacity, population_size, crossover_rate, mutation_rate, num_generations, tournament_size):
    population = generate_initial_population(len(items), population_size, capacity)
    best_fitness_evolution = []
    best_solution = population[0]  # Inicialize com o primeiro indivíduo da população
    best_fitness = fitness_function(best_solution, items, capacity)

    with tqdm(total=num_generations) as pbar:
        for gen in range(num_generations):
            print(f"Generation {gen}:")
            next_generation = []
            for _ in range(population_size):
                parent1 = tournament_selection(population, items, capacity, tournament_size)
                parent2 = tournament_selection(population, items, capacity, tournament_size)
                print(f"Selected parents: {parent1}, {parent2}")
                offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
                print(f"Offspring before mutation: {offspring1}, {offspring2}")
                mutate(offspring1, mutation_rate)
                mutate(offspring2, mutation_rate)
                print(f"Offspring after mutation: {offspring1}, {offspring2}")
                next_generation.extend([offspring1, offspring2])
            population = next_generation

            # Calcular o fitness da geração atual
            fitness_values = [fitness_function(individual, items, capacity) for individual in population]
            current_best_fitness = max(fitness_values)
            print(f"Best fitness in generation {gen}: {current_best_fitness}")
            best_fitness_evolution.append(current_best_fitness)

            # Atualizar a melhor solução encontrada
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution_index = fitness_values.index(best_fitness)
                best_solution = population[best_solution_index]

            pbar.update(1)

    return best_fitness_evolution, best_solution


# Função para gerar uma população inicial aleatória com proporção balanceada de itens selecionados e não selecionados
def generate_initial_population(size, population_size, capacity):
    population = []
    for _ in range(population_size):
        individual = [0] * size
        num_selected = random.randint(1, min(size, capacity))  # Limita o número máximo de itens selecionados ao tamanho total dos itens
        selected_indices = random.sample(range(size), num_selected)
        for index in selected_indices:
            individual[index] = 1
        population.append(individual)
    return population

# Função de fitness para calcular o valor total da mochila
def fitness_function(solution, items, capacity):
    total_value = sum(item.value for item, selected in zip(items, solution) if selected == 1)
    total_weight = sum(item.weight for item, selected in zip(items, solution) if selected == 1)
    excess_weight = max(0, total_weight - capacity)
    penalty = excess_weight / capacity if excess_weight > 0 else 0
    print(f"Fitness: {total_value * (1 - penalty)}, Total Value: {total_value}, Total Weight: {total_weight}, Excess Weight: {excess_weight}, Penalty: {penalty}")
    return total_value * (1 - penalty)

# Função para realizar a seleção por torneio
def tournament_selection(population, items, capacity, tournament_size):
    participants = random.sample(population, tournament_size)
    # Retorna o indivíduo com o maior fitness no torneio
    return max(participants, key=lambda x: fitness_function(x, items, capacity))

# Função para realizar o cruzamento (crossover)
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    # Se o crossover não ocorrer, retornar cópias completas dos pais
    return parent1[:], parent2[:]

# Função para realizar a mutação
def mutate(solution, mutation_rate):
    for j in range(len(solution)):
        if random.random() < mutation_rate:
            solution[j] = 1 - solution[j]  # Alterna entre 0 e 1

# Função para ler os itens do arquivo
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
            value = int(data[2])  # Agora estamos lendo o benefício como valor
            items.append(Item(weight, value))
    return items, capacity

if __name__ == "__main__":
    main()
