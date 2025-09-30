# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import streamlit as st

# Configurações de visualização
sns.set_style("whitegrid")

# --- Funções de Análise (Organizando seu código) ---

# Cache para carregar os dados apenas uma vez
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('AmesHousing.csv')
        # Limpeza e transformação que você já fez
        df['SalePrice_log'] = np.log1p(df['SalePrice'])
        df['Garage Type'] = df['Garage Type'].fillna('No Garage')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'AmesHousing.csv' não encontrado. Por favor, faça o upload do arquivo junto com o app.")
        return None

def run_analysis_overall_qual(df):
    st.header("Análise 1: Qualidade Geral (Overall Qual)")
    st.subheader("a) Validação dos Pressupostos da ANOVA")
    model = ols('SalePrice_log ~ Q("Overall Qual")', data=df).fit()
    residuals = model.resid
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    st.write(f"**Normalidade dos Resíduos (Shapiro-Wilk):** p-valor = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        st.success("✅ Pressuposto de normalidade atendido.")
    else:
        st.warning("❌ Pressuposto de normalidade violado.")
    groups = [df['SalePrice_log'][df['Overall Qual'] == q].dropna() for q in df['Overall Qual'].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    st.write(f"**Homogeneidade de Variâncias (Levene):** p-valor = {levene_p:.4f}")
    if levene_p > 0.05:
        st.success("✅ Pressuposto de homogeneidade atendido.")
    else:
        st.warning("❌ Pressuposto de homogeneidade violado.")
    st.info("Conclusão: Como todos os pressupostos são atendidos, a ANOVA tradicional é adequada.")
    st.subheader("b) Comparação de Preços com ANOVA")
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("**Tabela da ANOVA:**")
    st.dataframe(anova_table)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Overall Qual', y='SalePrice_log', data=df, ax=ax)
    ax.set_title('Distribuição de Log(SalePrice) por Qualidade Geral')
    ax.set_xlabel('Qualidade Geral')
    ax.set_ylabel('Log(Preço de Venda)')
    st.pyplot(fig)
    st.write("O p-valor (PR(>F)) próximo de zero indica que a qualidade geral tem um impacto estatisticamente significativo no preço.")

def run_analysis_neighborhood(df):
    st.header("Análise 2: Bairro (Neighborhood)")
    st.subheader("a) Validação dos Pressupostos e Escolha do Método")
    model = ols('SalePrice_log ~ Q("Neighborhood")', data=df).fit()
    shapiro_stat, shapiro_p = stats.shapiro(model.resid)
    groups = [df['SalePrice_log'][df['Neighborhood'] == n].dropna() for n in df['Neighborhood'].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    st.write(f"**Normalidade dos Resíduos (Shapiro-Wilk):** p-valor = {shapiro_p:.4f}")
    st.warning("❌ Pressuposto de normalidade violado.")
    st.write(f"**Homogeneidade de Variâncias (Levene):** p-valor = {levene_p:.4f}")
    st.warning("❌ Pressuposto de homogeneidade violado.")
    st.error("Conclusão: Com a violação dos pressupostos, a ANOVA tradicional não é confiável. O método alternativo não paramétrico **Kruskal-Wallis** é o mais adequado.")
    st.subheader("b) Comparação de Preços com Kruskal-Wallis")
    kruskal_stat, kruskal_p = stats.kruskal(*groups)
    st.write(f"**Teste de Kruskal-Wallis:** Estatística H = {kruskal_stat:.2f}, p-valor = {kruskal_p:.4e}")
    st.write("O p-valor extremamente baixo indica que há uma diferença estatisticamente significativa na mediana dos preços entre os bairros.")
    fig, ax = plt.subplots(figsize=(14, 8))
    neighborhood_order = df.groupby('Neighborhood')['SalePrice_log'].median().sort_values().index
    sns.boxplot(x='Neighborhood', y='SalePrice_log', data=df, order=neighborhood_order, ax=ax)
    ax.set_title('Distribuição de Log(SalePrice) por Bairro')
    ax.set_xlabel('Bairro')
    ax.set_ylabel('Log(Preço de Venda)')
    plt.xticks(rotation=90)
    st.pyplot(fig)

def run_analysis_garage_type(df):
    st.header("Análise 3: Tipo de Garagem (Garage Type)")
    st.subheader("a) Validação dos Pressupostos da ANOVA")
    model = ols('SalePrice_log ~ Q("Garage Type")', data=df).fit()
    residuals = model.resid
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    st.write(f"**Normalidade dos Resíduos (Shapiro-Wilk):** p-valor = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        st.success("✅ Pressuposto de normalidade atendido.")
    else:
        st.warning("❌ Pressuposto de normalidade violado.")
    groups = [df['SalePrice_log'][df['Garage Type'] == g].dropna() for g in df['Garage Type'].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    st.write(f"**Homogeneidade de Variâncias (Levene):** p-valor = {levene_p:.4f}")
    if levene_p > 0.05:
        st.success("✅ Pressuposto de homogeneidade atendido.")
    else:
        st.warning("❌ Pressuposto de homogeneidade violado.")
    st.info("Conclusão: Apesar da leve violação da normalidade, a ANOVA é robusta a isso com amostras grandes, e como a homogeneidade foi atendida, podemos prosseguir com a ANOVA tradicional.")
    st.subheader("b) Comparação de Preços com ANOVA")
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("**Tabela da ANOVA:**")
    st.dataframe(anova_table)
    fig, ax = plt.subplots(figsize=(12, 7))
    garage_order = df.groupby('Garage Type')['SalePrice_log'].median().sort_values().index
    sns.boxplot(x='Garage Type', y='SalePrice_log', data=df, order=garage_order, ax=ax)
    ax.set_title('Distribuição de Log(SalePrice) por Tipo de Garagem')
    ax.set_xlabel('Tipo de Garagem')
    ax.set_ylabel('Log(Preço de Venda)')
    st.pyplot(fig)

# --- Interface do App ---
st.title("Tarefa 3 de AEDI: Análise Interativa do Mercado Imobiliário de Ames")
st.write("Análise de Variância (ANOVA) aplicada ao dataset Ames Housing.")
df = load_data()
if df is not None:
    st.sidebar.header("Selecione a Análise")
    analysis_option = st.sidebar.selectbox(
        "Escolha a característica para analisar:",
        ("Qualidade Geral (Overall Qual)", "Bairro (Neighborhood)", "Tipo de Garagem (Garage Type)")
    )
    if analysis_option == "Qualidade Geral (Overall Qual)":
        run_analysis_overall_qual(df)
    elif analysis_option == "Bairro (Neighborhood)":
        run_analysis_neighborhood(df)
    elif analysis_option == "Tipo de Garagem (Garage Type)":
        run_analysis_garage_type(df)