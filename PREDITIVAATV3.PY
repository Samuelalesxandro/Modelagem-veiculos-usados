import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Carregar o arquivo diretamente do caminho fornecido
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS PREDITIVOS.xlsx"
df = pd.read_excel(file_path)

# 1) Filtrar valores inconsistentes na coluna KM usando Z-score
df = df[(np.abs((df["KM"] - df["KM"].mean()) / df["KM"].std()) < 3)]

# 2) Ajustar idade do veículo corretamente
df["idade do veiculo"] = 2024 - df["ANO"]

# 3) Criar variáveis dummies para CAMBIO
df["CAMBIO_Automatico"] = (df["CAMBIO"] == "AUTOMATICO").astype(int)

# 4) Padronizar KM e PREÇO com Min-Max Scaling
scaler = MinMaxScaler()
df[["KM", "PREÇO"]] = scaler.fit_transform(df[["KM", "PREÇO"]])

# Salvar o arquivo modificado
output_file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Modificações concluídas e arquivo salvo como {output_file_path}.")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Carregar dados
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df = pd.read_excel(file_path)

# Aplicar logaritmo em KM e PREÇO
df["log_KM"] = np.log1p(df["KM"])
df["log_PREÇO"] = np.log1p(df["PREÇO"])

# Converter a coluna CAMBIO em valores numéricos (Manual = 0, Automático = 1)
df["CAMBIO"] = df["CAMBIO"].map({"MANUAL": 0, "AUTOMATICO": 1})

# Criar variáveis dummies para MARCA e MODELO
df = pd.get_dummies(df, columns=["MARCA", "MODELO"], drop_first=True)

# Selecionar variáveis independentes e dependentes
X = df.drop(columns=["PREÇO", "KM", "log_PREÇO"])  # Excluímos PREÇO original e KM sem log
y = df["log_PREÇO"]

# Certificar-se de que X só contém valores numéricos
print("Colunas com tipo errado:", X.select_dtypes(include=['object']).columns)

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)  # Agora sem erro!

# Fazer previsões e avaliar desempenho
y_pred = model.predict(X_test)
print("Modelo treinado com sucesso!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df = pd.read_excel(file_path)

# Aplicar logaritmo em KM e PREÇO para melhor distribuição
df["log_KM"] = np.log1p(df["KM"])
df["log_PREÇO"] = np.log1p(df["PREÇO"])

# Converter CAMBIO para valores numéricos
df["CAMBIO"] = df["CAMBIO"].map({"MANUAL": 0, "AUTOMATICO": 1})

# Criar variáveis dummies para MARCA e MODELO
df = pd.get_dummies(df, columns=["MARCA", "MODELO"], drop_first=True)

# Selecionar variáveis independentes e dependente
X = df.drop(columns=["PREÇO", "KM", "log_PREÇO"])  # Excluímos valores brutos
y = df["log_PREÇO"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar modelo de Árvore de Decisão
tree_model = DecisionTreeRegressor(max_depth=5, criterion="squared_error", random_state=42)
tree_model.fit(X_train, y_train)

# Fazer previsões
y_pred = tree_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df = pd.read_excel(file_path)

# Aplicar logaritmo em KM e PREÇO para melhorar distribuição
df["log_KM"] = np.log1p(df["KM"])
df["log_PREÇO"] = np.log1p(df["PREÇO"])

# Converter CAMBIO para valores numéricos
df["CAMBIO"] = df["CAMBIO"].map({"MANUAL": 0, "AUTOMATICO": 1})

# Criar variáveis dummies para MARCA e MODELO
df = pd.get_dummies(df, columns=["MARCA", "MODELO"], drop_first=True)

# Selecionar variáveis independentes e dependente
X = df.drop(columns=["PREÇO", "KM", "log_PREÇO"])  # Excluímos valores brutos
y = df["log_PREÇO"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# Exibir importância das variáveis
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nImportância das variáveis:")
print(feature_importance.sort_values(ascending=False))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df = pd.read_excel(file_path)

# Aplicar logaritmo em KM e PREÇO para melhor distribuição
df["log_KM"] = np.log1p(df["KM"])
df["log_PREÇO"] = np.log1p(df["PREÇO"])

# Converter CAMBIO para valores numéricos
df["CAMBIO"] = df["CAMBIO"].map({"MANUAL": 0, "AUTOMATICO": 1})

# Criar variáveis dummies para MARCA e MODELO
df = pd.get_dummies(df, columns=["MARCA", "MODELO"], drop_first=True)

# Selecionar variáveis independentes e dependente
X = df.drop(columns=["PREÇO", "KM", "log_PREÇO"])  # Excluímos valores brutos
y = df["log_PREÇO"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar modelo de Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Fazer previsões
y_pred = gb_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Exibir importância das variáveis
feature_importance = pd.Series(gb_model.feature_importances_, index=X.columns)
print("\nImportância das variáveis:")
print(feature_importance.sort_values(ascending=False))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
file_path = r"C:\Users\Samue\OneDrive\Documents\MODELOS_PREDITIVOS_LIMPO.xlsx"
df = pd.read_excel(file_path)

# Aplicar logaritmo em KM e PREÇO para melhorar distribuição
df["log_KM"] = np.log1p(df["KM"])
df["log_PREÇO"] = np.log1p(df["PREÇO"])

# Converter CAMBIO para valores numéricos
df["CAMBIO"] = df["CAMBIO"].map({"MANUAL": 0, "AUTOMATICO": 1})

# Criar variáveis dummies para MARCA e MODELO
df = pd.get_dummies(df, columns=["MARCA", "MODELO"], drop_first=True)

# Selecionar variáveis independentes e dependente
X = df.drop(columns=["PREÇO", "KM", "log_PREÇO"])  # Excluímos valores brutos
y = df["log_PREÇO"]

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo XGBoost
xgb_model = XGBRegressor(random_state=42)

# Definir os hiperparâmetros a serem ajustados no Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

# Configurar Grid Search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1)

# Treinar Grid Search
grid_search.fit(X_train, y_train)

# Melhor combinação de hiperparâmetros
best_params = grid_search.best_params_
print(f"\nMelhores hiperparâmetros encontrados: {best_params}")

# Treinar modelo final com os melhores parâmetros
final_xgb_model = XGBRegressor(**best_params, random_state=42)
final_xgb_model.fit(X_train, y_train)

# Fazer previsões
y_pred = final_xgb_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nErro Quadrático Médio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Exibir importância das variáveis
feature_importance = pd.Series(final_xgb_model.feature_importances_, index=X.columns)
print("\nImportância das variáveis:")
print(feature_importance.sort_values(ascending=False))


