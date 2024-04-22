import random
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Importando tqdm para a barra de progresso

class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value

def main():
    # Leitura dos itens do arquivo
    items, capacity = read_items_from_file("./TRABALHO_PRATICO_1/instancias-mochila/KNAPDATA100.TXT")

    # Parâmetros da Busca Tabu
    tabu_size = 5
    max_iterations = 1000  # Reduzindo o número máximo de iterações

    # Chamada para a função que implementa a Busca Tabu
    solution, cost_evolution = tabu_search_knapsack(items, capacity, tabu_size, max_iterations)

    # Imprimir a solução encontrada
    print("Solução encontrada:")
    total_weight = 0
    total_value = 0
    for i, item in enumerate(items):
        if solution[i] == 1:
            total_weight += item.weight
            total_value += item.value
            print(f"Item {item.name}: Peso = {item.weight}, Valor = {item.value}")

    print(f"Peso total da mochila: {total_weight}")
    print(f"Valor total da mochila: {total_value}")

    # Imprimir a evolução da função de custo
    print("Evolução da função de custo:")
    for iteration, cost in enumerate(cost_evolution):
        print(f"Iteração {iteration+1}: {cost}")

    # Plotar o gráfico da evolução da função de custo
    plt.plot(cost_evolution)
    plt.xlabel('Iteração')
    plt.ylabel('Valor da Função de Custo')
    plt.title('Evolução da Função de Custo')
    plt.show()

def tabu_search_knapsack(items, capacity, tabu_size, max_iterations):
    current_solution = generate_random_solution(len(items))
    best_solution = copy.deepcopy(current_solution)
    current_cost = cost_function(current_solution, items, capacity)
    best_cost = current_cost
    tabu_list = []
    cost_evolution = [best_cost]

    # Usando tqdm para criar a barra de progresso
    with tqdm(total=max_iterations) as pbar:
        with ThreadPoolExecutor() as executor:
            for _ in range(max_iterations):
                neighbors = generate_neighbors(current_solution)
                futures = [executor.submit(evaluate_neighbor, neighbor, items, capacity) for neighbor in neighbors]
                results = [future.result() for future in futures]

                next_solution, next_cost = max(results, key=lambda x: x[1])

                if next_cost > best_cost:
                    best_solution = copy.deepcopy(next_solution)
                    best_cost = next_cost

                tabu_list.append(next_solution)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

                if next_cost != float('-inf'):  # Verifica se o custo é válido antes de adicionar à lista
                    cost_evolution.append(best_cost)

                current_solution = next_solution
                current_cost = next_cost
                
                # Atualizando a barra de progresso
                pbar.update(1)

    return best_solution, cost_evolution

def evaluate_neighbor(neighbor, items, capacity):
    neighbor_cost = cost_function(neighbor, items, capacity)
    return neighbor, neighbor_cost

def generate_random_solution(size):
    return [random.randint(0, 1) for _ in range(size)]

def cost_function(solution, items, capacity):
    total_value = 0
    total_weight = 0
    for i, selected in enumerate(solution):
        if selected == 1:
            total_value += items[i].value
            total_weight += items[i].weight

    if total_weight > capacity:
        # Retorna a diferença negativa entre o peso total e a capacidade
        return -(total_weight - capacity)

    return total_value

def generate_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        neighbor = list(solution)
        neighbor[i] = 1 - neighbor[i]  # Troca 0 por 1 e vice-versa
        neighbors.append(neighbor)
    return neighbors

def read_items_from_file(file_path):
    items = []
    capacity = 0
    with open(file_path, 'r') as file:
        lines = file.readlines()
        capacity = int(lines[0].strip())
        for line in lines[2:]:
            data = line.strip().split(',')
            name = data[0]
            weight = int(data[1])
            value = int(data[2])
            items.append(Item(name, weight, value))
    return items, capacity

if __name__ == "__main__":
    main()