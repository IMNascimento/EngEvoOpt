# Algoritmos Evolutivos: Comparação entre EP e CSA

Este repositório contém o código-fonte dos experimentos desenvolvidos para o artigo que compara o Evolutionary Programming (EP) e o Cuckoo Search Algorithm (CSA) em diferentes cenários de otimização, incluindo funções de benchmark e o problema de um redutor de velocidade com múltiplas restrições. Os experimentos foram realizados utilizando a linguagem Python, com a biblioteca Mealpy para a implementação dos algoritmos evolutivos.

## Estrutura do Repositório

- `src/exp_1.py`: Script para reproduzir o Experimento #1, que aplica os algoritmos EP e CSA em funções de benchmark como Griewank, Rastrigin e Weierstrass.
- `src/exp_2.py`: Script para reproduzir o Experimento #2, que aplica os algoritmos no problema de otimização do redutor de velocidade com múltiplas restrições.
- `src/requirements.txt`: Arquivo listando todas as dependências necessárias para executar os scripts.
- `results/`: Pasta onde estão os resultados obtidos após a execução dos experimentos do artigo desenvolvido.


## Pré-requisitos

Para reproduzir os experimentos, você precisará ter o Python 3.9+ instalado em seu ambiente. Recomenda-se utilizar um ambiente virtual para gerenciar as dependências.

## Instalação

1. Clone este repositório em sua máquina local:

    ```bash
    git clone https://github.com/IMNascimento/EngEvoOpt.git
    cd EngEvoOpt
    ```

2. Crie um ambiente virtual e ative-o:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    venv\Scripts\activate  # Para Windows
    ```

3. Instale as dependências listadas no `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Executando os Experimentos

### Experimento #1

O script src/exp_1.py permite reproduzir o primeiro experimento, que analisa a eficácia dos algoritmos EP e CSA em funções de benchmark com superfícies multimodais.

Para executar o experimento:

```bash
cd src
streamlit run exp_1.py
```

### Experimento #2

O script src/exp_2.py permite reproduzir o segundo experimento, que aplica os algoritmos ao problema de otimização do redutor de velocidade.

Para executar o experimento:

```bash
cd src
streamlit run exp_2.py
```

## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões de melhorias ou encontrar problemas, fique à vontade para abrir uma issue ou enviar um pull request.

##  Citação

Se você utilizar este código ou artigo em sua pesquisa ou projeto, por favor, cite-o da seguinte forma:

**Formato ABNT:**
NASCIMENTO, Igor M. *Análise Comparativa de Algoritmos de Otimização Evolutiva Aplicados a Problemas de Engenharia*. Disponível em:[imnascimento.github.io](https://imnascimento.github.io/Portifolio/assets/pdf/artigos/UFJF___An%C3%A1lise_Comparativa_de_Algoritmos_de_Otimiza%C3%A7%C3%A3o_Evolutiva_Aplicados_a_Problemas_de_Engenharia.pdf). Acesso em: 21/09/2024.

**BibTeX:**
```bibtex
@misc{EngEvoOpt,
  author = {Nascimento, Igor M.},
  title = {Análise Comparativa de Algoritmos de Otimização Evolutiva Aplicados a Problemas de Engenharia},
  year = {2024},
  howpublished = {\url{https://imnascimento.github.io/Portifolio/assets/pdf/artigos/UFJF___An%C3%A1lise_Comparativa_de_Algoritmos_de_Otimiza%C3%A7%C3%A3o_Evolutiva_Aplicados_a_Problemas_de_Engenharia.pdf}},
  note = {Acesso em: 21/09/2024}
}
```


## Contato

Para mais informações ou para acesso ao artigo, entre em contato através de [igor.muniz@estudante.ufjf.br](mailto:igor.muniz@estudante.ufjf.br).