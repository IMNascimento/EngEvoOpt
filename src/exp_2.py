import time
import numpy as np
from mealpy.evolutionary_based import EP
from mealpy.swarm_based import CSA 
from mealpy import FloatVar
from lib.apm import AdaptivePenaltyMethod
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

class APMOptimization:
    def __init__(self, number_of_constraints, variant="APM", model_class=EP.OriginalEP, num_executions=30, num_evaluations=10000):
        """
        Construtor.
        Parameters:
        - number_of_constraints: número de restrições do problema.
        - variant: variante do método de penalidade adaptativa. {APM, AMP_Med_3, AMP_Worst, APM_Spor_Mono}
        - model_class: classe do modelo de otimização (ex: EP da biblioteca Mealpy)
        - num_executions: número de execuções independentes para a otimização.
        - num_evaluations: número total de avaliações da função objetivo.
        """
        self.number_of_constraints = number_of_constraints
        self.variant = variant
        self.model_class = model_class
        self.num_executions = num_executions
        self.num_evaluations = num_evaluations
        self.apm = AdaptivePenaltyMethod(number_of_constraints, variant)

    def objective_function(self, solution):
        x1, x2, x3, x4, x5, x6, x7 = solution
        
        # Função objetivo (minimizar o peso W)
        W = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) \
            - 1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3)
        
        # Restrições do problema
        def g1(x):
            return 27 / (x1 * x2**2 * x3) - 1

        def g2(x):
            return 397.5 / (x1 * x2**2 * x3**2) - 1

        def g3(x):
            return 1.93 * x4**3 / (x2 * x3 * x6**4) - 1

        def g4(x):
            return 1.93 * x5**3 / (x2 * x3 * x7**4) - 1

        def g5(x):
            return np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) - 1100

        def g6(x):
            return np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) - 850

        def g7(x):
            return x2 * x3 - 40

        def g8(x):
            return 5 - x1 / x2

        def g9(x):
            return x1 / x2 - 12

        def g10(x):
            return 1.5 * x6 - x4

        def g11(x):
            return 1.1 * x7 - x5
        
        # Lista de restrições (g1 a g11)
        violations = [g1(solution), g2(solution), g3(solution), g4(solution), g5(solution), g6(solution), 
                    g7(solution), g8(solution), g9(solution), g10(solution), g11(solution)]
        
        # Penalização para violações (se violação > 0, é aplicada uma penalização)
        penalty = np.sum([violation**2 if violation > 0 else 0 for violation in violations])
        
        # Retorna o valor da função objetivo (W) + penalização para restrições
        return W , violations

    def penalized_objective_function(self, solution, population):
        """
        Função objetiva penalizada que calcula o fitness usando as penalidades adaptativas.
        Parameters:
        - solution: solução para avaliar
        - population: população para cálculo dos coeficientes de penalidade
        
        Returns:
        - fitness penalizado
        """
        V, violations = self.objective_function(solution)

        # Calcular valores da função objetivo e das violações de restrições para a população
        objective_values = np.zeros(len(population))
        constraint_violations = np.zeros((len(population), self.number_of_constraints))

        for i in range(len(population)):
            obj_val, viol = self.objective_function(population[i])
            objective_values[i] = obj_val
            constraint_violations[i] = viol[:self.number_of_constraints]  # Usar apenas as primeiras 3 restrições

        # Calcular os coeficientes de penalidade
        penalty_coefficients = self.apm.calculate_penalty_coefficients(objective_values, constraint_violations)

        # Calcular o fitness penalizado
        fitness = self.apm.calculate_single_fitness(V, violations[:self.number_of_constraints], penalty_coefficients)

        return fitness

    def run_optimization(self, lower_bounds, upper_bounds, pop_size=50):
        """
        Executa a otimização múltiplas vezes e retorna as métricas para o volume V.
        
        Parameters:
        - lower_bounds: limites inferiores para as variáveis de decisão.
        - upper_bounds: limites superiores para as variáveis de decisão.
        - pop_size: tamanho da população.
        
        Returns:
        - métricas de Melhor, Mediana, Média, Desvio Padrão e Pior para o volume V.
        """
        # Calcular o número de epochs com base no número total de avaliações e no tamanho da população
        epochs = self.num_evaluations // pop_size

        results = []
        for _ in range(self.num_executions):
            # Inicializar a população
            population = np.random.uniform(lower_bounds, upper_bounds, (pop_size, len(lower_bounds)))

            # Otimização usando a model_class passada com a função objetiva penalizada
            problem = {
                "obj_func": lambda solution: self.penalized_objective_function(solution, population),
                "bounds": FloatVar(lb=lower_bounds, ub=upper_bounds),
                "minmax": "min",
                "log_to": None,
            }

            model = self.model_class(epoch=epochs, pop_size=pop_size)
            model.solve(problem)

            best_solution = model.g_best.solution
            best_fitness = model.g_best.target.fitness

            # Armazenar os valores de V para a melhor solução
            V_best, _ = self.objective_function(best_solution)
            results.append(V_best)

        # Calcular as métricas
        melhor = np.min(results)
        mediana = np.median(results)
        media = np.mean(results)
        dp = np.std(results)
        pior = np.max(results)

        return melhor, mediana, media, dp, pior


def main():

    st.set_page_config(page_title="Otimização de EP e CSA com Restrições", page_icon="📊", layout="wide")

    # Parâmetros do problema
    number_of_constraints = 11 # Número de variáveis de dicsão, no caso do problema da mola são x1, x2 e x3

    variants = ["APM", "APM_Med_3", "APM_Worst", "APM_Spor_Mono"]
    model_classes = {"EP": EP.OriginalEP, "CSA": CSA.OriginalCSA}

    # Interface gráfica
    st.title("Otimização de EP e CSA com Restrições")
    st.sidebar.title("Configurações")

    with st.sidebar:
        with st.form(key="config_form"):
            num_executions = st.number_input("Número de execuções", min_value=1, max_value=100, value=35, step=1, key="num_executions")
            num_evaluations = st.number_input("Número total de avaliações", min_value=1000, max_value=100000, value=36000, step=1000, key="num_evaluations")
            pop_size = st.number_input("Tamanho da população", min_value=1, max_value=200, value=50, step=1, key="pop_size")
            submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return
    
    # Limites das variáveis de decisão
    lower_bounds = [2.6, 0.7, 17, 7.3, 7.8, 2.9, 2.9]  # [x1, x2, x3, x4, x5, x6, x7]
    upper_bounds = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 3.9]  # [x1, x2, x3, x4, x5, x6, x7]

    resultados = []
    col1, col2 = st.columns(2)

    # Percorrer cada algoritmo de otimização e variante do APM
    with st.spinner("Executando otimizações..."):
        start_time = time.time()
        for key in model_classes.keys():
            for variant in variants:
                optimizer = APMOptimization(
                    number_of_constraints=number_of_constraints,
                    variant=variant,
                    model_class=model_classes[key],
                    num_executions=num_executions,
                    num_evaluations=num_evaluations
                )
                # Executar a otimização e obter as métricas
                melhor, mediana, media, dp, pior = optimizer.run_optimization(lower_bounds, upper_bounds, pop_size=pop_size)
                resultados.append((key, variant, melhor, mediana, media, dp, pior))
        end_time = time.time()
        # Calcular horas, minutos e segundos que foram necessários para a execução
        tempo_execucao = end_time - start_time
        horas = int(tempo_execucao // 3600)
        minutos = int((tempo_execucao % 3600) // 60)
        segundos = int(tempo_execucao % 60)
        

    st.success(f"Execução finalizada em {horas} horas, {minutos} minutos e {segundos} segundos.")  

    # Criar dataframe com os resultados
    df = pd.DataFrame(resultados, columns=["Algoritmo", "Variante", "Melhor", "Mediana", "Média", "Desvio Padrão", "Pior"])

    # Divindindo o Dataframe por algoritmo
    df_ep = df[df["Algoritmo"] == "EP"]
    df_ep = df_ep.drop(columns=["Algoritmo"])
    df_csa = df[df["Algoritmo"] == "CSA"]
    df_csa = df_csa.drop(columns=["Algoritmo"])


    col1.write("Resultados para o EP")
    col1.write(df_ep)
    # Gráfico de barras para cada algoritmo
    fig_ep = px.bar(df_ep, x="Variante", y="Melhor", color="Variante", title="Melhor valor de V para cada variante do EP")
    col1.plotly_chart(fig_ep)

    
    col2.write("Resultados para o CSA")
    col2.write(df_csa)
    fig_csa = px.bar(df_csa, x="Variante", y="Melhor", color="Variante", title="Melhor valor de V para cada variante do CSA")
    col2.plotly_chart(fig_csa)

    # Gráfico de barras unificado
    fig = px.bar(df, x="Algoritmo", y="Melhor", color="Variante", barmode="group", title="Melhor valor de V para cada algoritmo e variante")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()