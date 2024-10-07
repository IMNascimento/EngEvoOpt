import streamlit as st
import numpy as np
import plotly.express as px
from mealpy.evolutionary_based.EP import OriginalEP as BaseEP
from mealpy.swarm_based.CSA import OriginalCSA as BaseCSA
from opfunu.cec_based import F52017, F92017, F112017
from mealpy import FloatVar
import pandas as pd
import time

# Função de Penalização - Eq. (4)
def calculate_penalty(f, constraints, population):
    """
    Aplica a estratégia de penalização da Eq. (4).
    Penaliza soluções inviáveis com base nas violações de restrições.
    """
    # Verificar se a solução atual é viável
    if all(constraint <= 0 for constraint in constraints):
        return f  # Solução viável, retorna o valor da função objetivo diretamente
    else:
        # Penalização: soma das violações de restrições
        violation_sum = sum(max(0, constraint) for constraint in constraints)

        # Encontrar o fmax (pior solução viável na população)
        feasible_solutions = []
        for sol in population:
            # Recalcula as restrições para cada solução
            _, sol_constraints = objective_function(sol)
            if all(constraint <= 0 for constraint in sol_constraints):
                feasible_solutions.append(sol)
        
        if feasible_solutions:
            # Se houver soluções viáveis, encontra a de pior aptidão
            fmax = max(objective_function(sol)[0] for sol in feasible_solutions)
        else:
            fmax = f  # Se não houver soluções viáveis, usa o próprio f como fallback
        
        # Retornar a penalização
        return fmax + violation_sum

f1 = F52017(ndim=30, f_bias=0, f_shift='shift_data_2', f_matrix='M_2_D')
f2 = F92017(ndim=30, f_bias=0, f_shift='shift_data_2', f_matrix='M_2_D')
f3 = F112017(ndim=30, f_bias=0, f_shift='shift_data_2', f_matrix='M_2_D')

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": None
}

p2 = {
    "bounds": FloatVar(lb=f2.lb, ub=f2.ub),
    "obj_func": f2.evaluate,
    "minmax": "min",
    "name": "F9",
    "log_to": None
}

p3 = {
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "obj_func": f3.evaluate,
    "minmax": "min",
    "name": "F11",
    "log_to": None
}

resultados = {
    "Problema": [],
    "unshifted": [],
    "shifted": [],
    "ratio": []
}

algoritmos = ["EP", "CSA"]

def objective_function(solution):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30 = solution
    # Suponha que apenas as primeiras 11 variáveis são relevantes
    #x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = solution[:11]
    
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
    
    # Retorna o valor da função objetivo (W) + as violações de restrições
    return W, violations

# Função de execução do algoritmo com penalização
def run_algorithm(algoritmo, problem, pop_size, epoch):
    if algoritmo == "EP":
        model = BaseEP(epoch=epoch, pop_size=pop_size)
    elif algoritmo == "CSA":
        model = BaseCSA(epoch=epoch, pop_size=pop_size)
    else:
        raise ValueError("Algoritmo não suportado")

    model.solve(problem)

    # Fitness original da melhor solução
    best_fitness = np.abs(model.g_best.target.fitness)

    # Calcular as restrições manualmente usando a função 'objective_function'
    best_solution = model.g_best.solution
    _, constraints = objective_function(best_solution)

    # Aplicar a penalização usando a melhor solução
    fitness_with_penalty = calculate_penalty(best_fitness, constraints, [best_solution])

    return fitness_with_penalty

def run_experiment(algoritmo, problem, pop_size, epoch, use_shift):
    results = []
    for function_choice, problem in zip(["F5", "F9", "F11"], [p1, p2, p3]):
        best_fitness = run_algorithm(algoritmo, problem, pop_size, epoch)
        results.append(best_fitness)

        if use_shift:
            if function_choice in resultados["Problema"]:
                idx = resultados["Problema"].index(function_choice)
                if not isinstance(resultados["shifted"][idx], list):
                    resultados["shifted"][idx] = [resultados["shifted"][idx]]
                resultados["shifted"][idx].append(best_fitness)
            else:
                resultados["Problema"].append(function_choice)
                resultados["unshifted"].append(0.0)
                resultados["shifted"].append(best_fitness)
                resultados["ratio"].append(1.0)
        else:
            if function_choice in resultados["Problema"]:
                idx = resultados["Problema"].index(function_choice)
                if not isinstance(resultados["unshifted"][idx], list):
                    resultados["unshifted"][idx] = [resultados["unshifted"][idx]]
                resultados["unshifted"][idx].append(best_fitness)
            else:
                resultados["Problema"].append(function_choice)
                resultados["unshifted"].append(best_fitness)
                resultados["shifted"].append(0.0)
                resultados["ratio"].append(1.0)

def main():

    st.set_page_config(page_title="Otimização com EP e CSA", page_icon="🧊", layout='wide')

    st.title("Otimização com EP e CSA utilizando Mealpy")
    
    col1, col2 = st.columns(2)
    
    with st.sidebar:
        with st.form(key="config_form"):
            nruns = st.number_input("Número de Execuções", min_value=1, max_value=50, value=20)
            maxfes = st.number_input("Número de Avaliações de Função", min_value=1000, max_value=100000, value=50000)

            pop_size = st.number_input("Tamanho da população", min_value=1, max_value=200, value=50, step=1, key="pop_size")
            epoch = round(maxfes / pop_size)
            submit_button = st.form_submit_button("Executar")

    if not submit_button:
        return

    with col1:
        start_time = time.time()
        agrupados = {
            "dataframes": [],
            "geomeans": []
        }

        for algoritmo in algoritmos:
            with st.spinner(f"Executando otimizações para {algoritmo}..."):
                for run in range(nruns):
                    run_experiment(algoritmo, p1, pop_size, epoch, False)
                    run_experiment(algoritmo, p1, pop_size, epoch, True)

            end_time = time.time()

            # Calcular horas, minutos e segundos que foram necessários para a execução
            tempo_execucao = end_time - start_time
            horas = int(tempo_execucao // 3600)
            minutos = int((tempo_execucao % 3600) // 60)
            segundos = int(tempo_execucao % 60)
            
            # Atualizar o ratio para cada função
            for idx in range(len(resultados["Problema"])):
                resultados["shifted"][idx] = np.mean(resultados["shifted"][idx])
                resultados["unshifted"][idx] = np.mean(resultados["unshifted"][idx])
                resultados["ratio"][idx] = resultados["unshifted"][idx] / resultados["shifted"][idx]
            
            with col1:
                st.write(f"Resultados para o algoritmo {algoritmo}")
                
                st.write("Média geométrica da razão entre as funções com e sem shift")
                geometric_mean = np.prod(resultados["ratio"]) ** (1 / len(resultados["ratio"]))
                
                df = pd.DataFrame(resultados)
                df["unshifted"] = df["unshifted"].apply(lambda x: f"{x:.2e}")
                df["shifted"] = df["shifted"].apply(lambda x: f"{x:.2e}")
                df["ratio"] = df["ratio"].apply(lambda x: f"{x:.2e}")
                
                df.insert(0, "Algoritmo", algoritmo)
                
                st.write(df)
                agrupados["dataframes"].append(df)

                st.write(f"Média Geométrica (ratio): {geometric_mean:.2e}")
                agrupados["geomeans"].append([algoritmo, geometric_mean])

                st.success(f"Execução finalizada em {horas} horas, {minutos} minutos e {segundos} segundos.")
            
    with col2:
        df_unificado = pd.concat(agrupados["dataframes"])
        colors = {"EP": "dodgerblue", "CSA": "goldenrod"}

        fig = px.bar(df_unificado, x="Problema", y="ratio", title=f"Razão entre as funções com e sem shift para os algoritmos EP e CSA", text="ratio", color="Algoritmo", labels={"ratio": "Razão", "Problema": "Função"}, barmode="group", color_discrete_map=colors)
        fig.update_yaxes(tickformat=".2e")
        st.plotly_chart(fig)

        medias = pd.DataFrame(agrupados["geomeans"], columns=["Algoritmo", "geomeans"])

        fig = px.bar(medias, x="Algoritmo", y="geomeans", title="Média Geométrica das razões entre as funções com e sem shift para os algoritmos EP e CSA", color="Algoritmo", text="geomeans", labels={"Algoritmo": "Algoritmo", "geomeans": "Média Geométrica"}, color_discrete_map=colors)
        fig.update_yaxes(tickformat=".2e")
        fig.update_traces(texttemplate="%{text:.2e}")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
