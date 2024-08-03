# Primeira célula: Instalação das bibliotecas necessárias
# Execute isso no terminal
# pip install crewai
# pip install langchain
# pip install openai
# pip install kaggle
# pip install pandas
# pip install matplotlib
# pip install seaborn

# Segunda célula: Importar as bibliotecas e configurar as chaves de API
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar as chaves de API
os.environ["OPENAI_API_KEY"] = ""
os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""

# Terceira célula: Definir os agentes
# Definir o agente analista de negócio
business_analyst = Agent(
    role='Business Analyst',
    goal='Analyze the report data',
    verbose=True,
    memory=True,
    backstory='Experienced in business data analysis and reporting.',
    tools=[]  # Nenhuma ferramenta adicional configurada
)

# Definir o agente analista de dados
data_analyst = Agent(
    role='Data Analyst',
    goal='Generate a report with charts based on data',
    verbose=True,
    memory=True,
    backstory='Skilled in data visualization and reporting.',
    tools=[]  # Nenhuma ferramenta adicional configurada
)

# Definir o agente cientista de dados
data_scientist = Agent(
    role='Data Scientist',
    goal='Generate executable code to analyze the Kaggle dataset',
    verbose=True,
    memory=True,
    backstory='Expert in data analysis and machine learning.',
    tools=[]  # Nenhuma ferramenta adicional configurada
)

# Quarta célula: Definir as tarefas
# Tarefa para o analista de negócio
business_analysis_task = Task(
    description='Analyze the data from the report.',
    expected_output='A detailed analysis report.',
    agent=business_analyst
)

# Tarefa para o analista de dados
data_analysis_task = Task(
    description='Generate a report with charts based on the data provided by the data scientist.',
    expected_output='A comprehensive report with visualizations.',
    agent=data_analyst
)

# Tarefa para o cientista de dados
data_science_task = Task(
    description='Generate executable code to analyze the Kaggle dataset: https://www.kaggle.com/datasets/yasserh/walmart-dataset?resource=download',
    expected_output='An executable Python script for data analysis.',
    agent=data_scientist
)

# Quinta célula: Configurar e executar o Crew
# Definir a equipe com os agentes e tarefas
crew = Crew(
    agents=[business_analyst, data_analyst, data_scientist],
    tasks=[business_analysis_task, data_analysis_task, data_science_task],
    process=Process.sequential,
    name='Data Analysis Crew',
    goal='Analyze data from a Kaggle dataset'
)

# Iniciar a execução da equipe
inputs = {
    'dataset_url': 'https://www.kaggle.com/datasets/yasserh/walmart-dataset?resource=download',
    'analysis_goal': 'Analyze the sales data of Walmart to identify trends and generate insights.'
}
result = crew.kickoff()

print(result)

# Sexta célula: Adicional - Código para baixar e carregar os dados do Kaggle
# Execute isso no terminal
# !kaggle datasets download -d yasserh/walmart-dataset

# Descompactar o arquivo baixado (use um programa como 7-Zip ou WinRAR)

# Carregar os dados do arquivo Walmart.csv
walmart_data = pd.read_csv('./Walmart.csv')

# Exibir as primeiras linhas do dataframe
print(walmart_data.head())

# Análise de dados e geração de gráficos
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))

# Exemplo de gráfico: Total de vendas por loja
sales_per_store = walmart_data.groupby('Store')['Weekly_Sales'].sum().reset_index()
sns.barplot(x='Store', y='Weekly_Sales', data=sales_per_store)
plt.title('Total Weekly Sales per Store')
plt.xlabel('Store')
plt.ylabel('Total Weekly Sales')
plt.show()

# Exemplo de gráfico: Vendas semanais ao longo do tempo
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Weekly_Sales', data=walmart_data)
plt.title('Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.xticks(rotation=45)
plt.show()
