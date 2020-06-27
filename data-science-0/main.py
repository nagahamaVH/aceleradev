#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    # Retorne aqui o resultado da questão 2.
    black_friday["Age"].unique()
    return black_friday.query("Gender == 'F' & Age == '26-35'").shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[5]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday["User_ID"].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[6]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[7]:


def q5():
    # Retorne aqui o resultado da questão 5.
    df = black_friday.isna().any(1)
    return df.sum() / len(df)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[8]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return black_friday.isna().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[9]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday["Product_Category_3"].mode()[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[10]:


def q8():
    # Retorne aqui o resultado da questão 8.
    def min_max_normalization(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    return min_max_normalization(black_friday["Purchase"]).mean()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[11]:


def q9():
    # Retorne aqui o resultado da questão 9.
    def standardize(x):
        return (x - np.mean(x)) / np.std(x)
    
    black_friday["Standardized_Purchase"] = standardize(black_friday["Purchase"])
    return black_friday.query("Standardized_Purchase <= 1 & Standardized_Purchase >=-1").shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[12]:


def q10():
    # Retorne aqui o resultado da questão 10.
    df = black_friday[["Product_Category_2", "Product_Category_3"]].isna()
    return df.query("Product_Category_2 == True")["Product_Category_3"].all()

