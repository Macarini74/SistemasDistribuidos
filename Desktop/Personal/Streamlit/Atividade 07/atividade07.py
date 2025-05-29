import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

df = pd.read_csv('Atividade 07/Life_Expectancy_Data.csv', encoding='latin1')
df_copy = df.copy()
df_copy.columns = df_copy.columns.str.strip()
df_cleaned = df_copy.dropna(subset='Life expectancy')

coluna = st.sidebar.selectbox('Escolha uma opção:', ['Home','Questão 01','Questão 02', 'Questão 03', 'Questão 04',
                                                     'Questão 05', 'Questão 06', 'Questão 07', 'Questão 08'])

# 1. Os vários fatores de previsão inicialmente escolhidos realmente afetam a expectativa de
# vida? Quais são as variáveis de previsão que realmente afetam a expectativa de vida?

if coluna == 'Home':
    st.title('Análise de Dados OMS')
    st.subheader('Dataframe Completo')
    st.write(df)
elif coluna == 'Questão 01':

    st.subheader('Avaliando quais variáveis de previsão realmente afetam a expectativa de vida')
    # Remover colunas não numéricas

    df_copy = df_copy.drop(columns=["Country", "Year", "Status"])
    df_copy = df_copy.dropna()

    #Separa Features (Variáveis que Afetam) e Target (Life Expectancy)
    x = df_copy.drop('Life expectancy', axis=1)
    y = df_copy['Life expectancy']

    # Dividir dados entre teste e treino
    x_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    # Treinando o modelo Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # Importância das Features
    importances = model.feature_importances_
    features = x.columns

    # DataFrame para Visualização
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot do modelo utilizando gráficos de barras
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_importance, ax=ax, palette='viridis')
    ax.set_title('Variáveis que Afetam a Espectativa de Vida')
    st.pyplot(fig)

    st.markdown('##### Podemos observar através do gráfico que a variável que mais afeta a expectativa de vida é:' \
    '*Income composition of Resources*')

    inc_comp_res_country = df.groupby('Country')['Income composition of resources'].sum()
    st.markdown('##### Soma dos valores de Income Composition of Resources por país')
    st.write(inc_comp_res_country)

elif coluna == 'Questão 02':
# 2. Um país com menor expectativa de vida (<65) deve aumentar seus gastos com saúde
# para melhorar sua expectativa de vida média?

    df_copy = df_copy.dropna(subset=['Life expectancy'])
    df_copy['Life expectancy'] = df_copy['Life expectancy'].astype(int)
    df_copy['low_life_expectancy'] = df_copy["Life expectancy"] < 65
    df_copy['high_life_expectancy'] = df_copy["Life expectancy"] >= 65

    fig, ax = plt.subplots()
    sns.scatterplot(df_copy,
                    x='Total expenditure',
                    y='Life expectancy',
                    hue='low_life_expectancy',
                    color='coolwarm',
                    ax=ax
                    )
    ax.set_title("Gastos com Saúde vs Expectativa de Vida")
    st.pyplot(fig)

    cor = df_copy[['low_life_expectancy', 'Total expenditure', 'percentage expenditure']].corr()
    cor2 = df_copy[['high_life_expectancy', 'Total expenditure', 'percentage expenditure']].corr()
    
    fig2, axs = plt.subplots(1,2, figsize=(16,5))
    sns.heatmap(cor, cmap='coolwarm', ax=axs[0], annot=True, square=True, fmt='.2f')
    
    sns.heatmap(cor2, cmap='coolwarm', ax=axs[1], annot=True, square=True, fmt='.2f')
    st.pyplot(fig2)
    
    st.markdown('Como pode se observar pelos gráficos (heatmaps) acima, países que '\
                'possuem altos gastos com saúde tendem a ter uma expectativa de vida '\
                'maior ou igual a 65.\n' \
                )

elif coluna == 'Questão 03':
    #3. Como as taxas de mortalidade infantil e adulta afetam a expectativa de vida?
    cor = df_copy[['Life expectancy', 'Adult Mortality', 'infant deaths']].corr()
    st.markdown('### Parâmetros:\n' \
            '* Se negativo: indica que quanto maior a mortalidade, menor a expectativa de vida.\n'
            '* Se positivo: indica que quanto menor a mortalidade, maior a expectativa de vida.\n'
            )
    fig, ax = plt.subplots()
    sns.heatmap(cor, cmap='coolwarm', ax=ax, annot=True, square=True, fmt='.2f')
    st.pyplot(fig)

    fig2, axs = plt.subplots(1,2, figsize=(12,5))
    
    sns.scatterplot(data=df_copy, x="Life expectancy", y="infant deaths", ax=axs[0])
    axs[0].set_title("Mortalidade Infantil vs Expectativa de Vida")

    sns.scatterplot(data=df_copy, x="Life expectancy", y="Adult Mortality", ax=axs[1])
    axs[1].set_title("Mortalidade Adulta vs Expectativa de Vida")

    st.pyplot(fig2)

elif coluna == 'Questão 04':
#     4. A expectativa de vida tem correlação positiva ou negativa com hábitos alimentares, estilo
#     de vida, exercícios, fumo, consumo de álcool etc.
    cor = df_copy[['Life expectancy', 'Alcohol', 'BMI']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(cor, cmap='crest', ax=ax, annot=True, square=True, fmt='.2f')
    st.pyplot(fig)

elif coluna == 'Questão 05':
    # 5. Qual é o impacto da escolaridade na expectativa de vida dos seres humanos?
    
    fig = px.scatter(
        df_cleaned,
        x='Life expectancy',
        y='Schooling',
        color='Country',
        size='Life expectancy',
        template='simple_white',
        opacity=0.7,
        title='<b> Expectativa de vida por Escolaridade'
    )
    st.plotly_chart(fig)

    st.write('Países com nível de escolaridade maior, tendem a ter a expectiva de vida maior.')

elif coluna == 'Questão 06':
    # 6. A expectativa de vida tem relação positiva ou negativa com o consumo de álcool?
    cor = df_copy[['Life expectancy', 'Alcohol']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(cor, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, annot=True, square=True, fmt='.2f')
    st.pyplot(fig)

elif coluna == 'Questão 07':
    # 7. Países densamente povoados tendem a ter menor expectativa de vida?
    
    cor = df_copy[['Population', 'Life expectancy']].corr()
    st.write(cor)

    fig = px.scatter(
        df_cleaned,
        x='Life expectancy',
        y='Population',
        color='Country',
        size='Life expectancy',
        template='simple_white',
        opacity=0.81,
        title='<b> Expectativa de Vida por População'
    )   
    st.plotly_chart(fig)
else:
    # 8. Qual é o impacto da cobertura de imunização na expectativa de vida?
    cor = df_copy[['Life expectancy', 'Hepatitis B', 'Polio', 'Diphtheria']].corr()

    fig, ax = plt.subplots()
    sns.heatmap(cor, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax, annot=True, square=True, fmt='.2f')
    st.pyplot(fig)

    st.write('Essas relações positivas indicam que maior cobertura vacinal está associada a maior expectativa de vida.')

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(data=df_copy, x='Hepatitis B', y='Life expectancy', ax=axs[0])
    axs[0].set_title("Hepatitis B vs Expectativa de Vida")

    sns.scatterplot(data=df_copy, x='Polio', y='Life expectancy', ax=axs[1])
    axs[1].set_title("Polio vs Expectativa de Vida")

    sns.scatterplot(data=df_copy, x='Diphtheria', y='Life expectancy', ax=axs[2])
    axs[2].set_title("Difteria vs Expectativa de Vida")

    st.pyplot(fig)

    st.markdown('A análise indica que há uma associação positiva entre cobertura de ' \
                'imunização e expectativa de vida. Ou seja, países com maior cobertura vacinal'
                ' (contra Hepatite B, Polio e Difteria) tendem a apresentar maior longevidade da população.' \
                ' Isso reforça a importância de políticas públicas voltadas à vacinação')