import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap

phi = 1.618
# Cargar los datos desde el archivo CSV
data = pd.read_csv('credit_risk_dataset.csv')

# Eliminar filas con valores NaN en 'loan_int_rate'
data = data.dropna()

categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
le = LabelEncoder()
for i in categorical_features:
    data[i] = le.fit_transform(data[i])

# Lista de regresores (excluyendo 'loan_status' y 'loan_int_rate')
regressors_names = ['Age', 'Income', 'Employment Length', 'Loan Amount', 'Interest Rate', 'Percent Income', 'Credit History Length']
regressors = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Iterar sobre cada regresor para generar los gráficos
for i in range(len(regressors)):
    regressor = regressors[i]

    percentil_5 = data[regressor].quantile(0.05)
    percentil_95 = data[regressor].quantile(0.95)
    data_5_95 = data[(data[regressor] >= percentil_5) & (data[regressor] <= percentil_95)]

    # Ordenar el conjunto de datos por la tasa de interés
    data_sorted = data_5_95.sort_values(by=regressor)

    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=(3 * phi, 3))

    # Graficar la distribución de la tasa de interés en el eje principal con el cmap red_blue
    n, bins, patches = ax1.hist(data_sorted[regressor], bins=30, edgecolor='k', alpha=0.7)
    ##########################

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', shap.plots.colors.red_blue(c))

    ##############################
    ax1.set_xlabel(regressors_names[i])
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{regressors_names[i]} Distribution',  fontweight='bold')
    ax1.grid(True)

    plt.grid(False)
    plt.tight_layout()
    plt.show()


########################################################################################
#Percentiles with DR
########################################################################################




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('credit_risk_dataset.csv')
# Eliminar filas con valores NaN en 'loan_int_rate'
data = data.dropna()

# Lista de características continuas
regressors_names = ['Age', 'Income', 'Employment Length', 'Loan Amount', 'Interest Rate', 'Percent Income', 'Credit History Length']
regressors = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Número total de buenos (préstamos no en mora)
total_buenos = len(data[data['loan_status'] == 0])

# Iterar sobre cada característica continua para generar los gráficos
for i in range(len(regressors)):
    regressor = regressors[i]
    percentil_5 = data[regressor].quantile(0.05)
    percentil_95 = data[regressor].quantile(0.95)
    data_5_95 = data[(data[regressor] >= percentil_5) & (data[regressor] <= percentil_95)]

    # Ordenar el conjunto de datos por la tasa de interés
    data_sorted = data_5_95.sort_values(by=regressor)

    # Calcular los deciles de la tasa de interés
    deciles = data_sorted[regressor].quantile([0.1 * i for i in range(1, 10)])

    # Añadir el mínimo y máximo para los bordes de los bins
    bins = [data_sorted[regressor].min()] + list(deciles) + [data_sorted[regressor].max()]

    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=(3 * phi, 3))

    # Graficar la distribución de la tasa de interés
    counts, bins, patches = ax1.hist(data_sorted[regressor], bins=bins, edgecolor='k', alpha=0.7)
    ax1.set_xlabel(regressors_names[i])
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{regressors_names[i]} Percentiles and DR', fontweight='bold')


    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    print(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', shap.plots.colors.red_blue(c))


    # Calcular la TMO para cada bin
    # Calcular la TMO para cada bin
    TMO = []
    for j in range(len(bins) - 1):
        bin_mask = (data_sorted[regressor] >= bins[j]) & (data_sorted[regressor] < bins[j + 1])
        bin_data = data_sorted[bin_mask]
        if len(bin_data) > 0:
            num_malos = bin_data['loan_status'].sum()  # Asumiendo que 'loan_status' indica malos con 1 y buenos con 0
            num_acreditados = len(bin_data)
            tmo = num_malos / num_acreditados
        else:
            tmo = 0
        TMO.append(tmo)


    TMO = np.array(TMO)
    ind = np.where(TMO != 0)[0]

    TMO = TMO[ind]
    bins = bins[ind]

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_centers = np.append(bin_centers, 0.5 * (bins[-1] + percentil_95))
    # Crear el segundo eje para la TMO y TMO ajustada
    ax2 = ax1.twinx()

    # Añadir la curva de TMO a la gráfica en el segundo eje
    #ax2.plot(bins[:-1], TMO, color='red', marker='o', label='TMO', linewidth=2)
    ax2.plot(bin_centers, TMO, color='red', marker = 'o', label='DR', linewidth=2)
    ax2.set_ylabel('Default Rate (DR)')

    # Añadir leyendas
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()




############################################################################
#Percentiles and Adjusted DR
############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('credit_risk_dataset.csv')
# Eliminar filas con valores NaN en 'loan_int_rate'
data = data.dropna()

# Lista de características continuas
regressors_names = ['Age', 'Income', 'Employment Length', 'Loan Amount', 'Interest Rate', 'Percent Income', 'Credit History Length']
regressors = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Número total de buenos (préstamos no en mora)
total_buenos = len(data[data['loan_status'] == 0])

# Iterar sobre cada característica continua para generar los gráficos
for i in range(len(regressors)):
    regressor = regressors[i]
    percentil_5 = data[regressor].quantile(0.05)
    percentil_95 = data[regressor].quantile(0.95)
    data_5_95 = data[(data[regressor] >= percentil_5) & (data[regressor] <= percentil_95)]

    # Ordenar el conjunto de datos por la tasa de interés
    data_sorted = data_5_95.sort_values(by=regressor)

    # Calcular los deciles de la tasa de interés
    deciles = data_sorted[regressor].quantile([0.1 * i for i in range(1, 10)])

    # Añadir el mínimo y máximo para los bordes de los bins
    bins = [data_sorted[regressor].min()] + list(deciles) + [data_sorted[regressor].max()]

    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=(3 * phi, 3))

    # Graficar la distribución de la tasa de interés
    counts, bins, patches = ax1.hist(data_sorted[regressor], bins=bins, edgecolor='k', alpha=0.7)
    ax1.set_xlabel(regressors_names[i])
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{regressors_names[i]} Percentiles and Adjusted DR', fontweight='bold')


    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)
    print(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', shap.plots.colors.red_blue(c))


    # Calcular la TMO para cada bin
    TMO = []
    ajuste = []
    for i in range(len(bins) - 1):
        bin_mask = (data_sorted[regressor] >= bins[i]) & (data_sorted[regressor] < bins[i + 1])
        bin_data = data_sorted[bin_mask]
        if len(bin_data) > 0:
            tmo = bin_data['loan_status'].mean()  # Proporción de mora en el bin
            buenos_bin = len(bin_data[bin_data['loan_status'] == 0])
            ajuste.append(tmo / (buenos_bin / total_buenos))  # TMO ajustada
        else:
            tmo = 0
            ajuste.append(0)
        TMO.append(tmo)


    ajuste = np.array(ajuste)
    ind = np.where(ajuste != 0)[0]

    ajuste = ajuste[ind]
    bins = bins[ind]

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_centers = np.append(bin_centers, 0.5 * (bins[-1] + percentil_95))
    # Crear el segundo eje para la TMO y TMO ajustada
    ax2 = ax1.twinx()

    # Añadir la curva de TMO a la gráfica en el segundo eje
    #ax2.plot(bins[:-1], TMO, color='red', marker='o', label='TMO', linewidth=2)
    ax2.plot(bin_centers, ajuste, color='blue', marker = 'o', label='Adjusted DR', linewidth=2)
    ax2.set_ylabel('Adjusted DR')

    # Añadir leyendas
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



############################################################################
#Categorical Ds¡istributions with DR
############################################################################



switcher = {
    'DEBTCONSOLIDATION': 'DEBT',
    'HOMEIMPROVEMENT': 'HOME',
    'EDUCATION': 'EDUC.',
    'MEDICAL': 'MED.',
    'PERSONAL': 'PERS.',
    'VENTURE': 'VENT.'
}
def switch(x):
  return switcher.get(x, x)

  switcher2 = {
    'MORTGAGE': 'MORTG.'
}

def switch2(x):
  return switcher2.get(x, x)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('credit_risk_dataset.csv')
data['loan_intent'] = list(map(switch, data['loan_intent']))
data['person_home_ownership'] = list(map(switch2, data['person_home_ownership']))
# Eliminar filas con valores NaN en 'loan_int_rate'
data = data.dropna(subset=['loan_int_rate'])

# Lista de características categóricas
regressors_names = ['Home Ownership', 'Loan Intent', 'Loan Grade', 'Historical Default']
regressors = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Iterar sobre cada característica categórica para generar los gráficos
for i in range(len(regressors)):
    regressor = regressors[i]
    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=(3 * phi, 3))

    count = len(np.unique(data[regressor]))
    colors = [shap.plots.colors.red_blue((j) / (count - 1)) for j in range(count)]

    # Graficar el histograma de la característica categórica
    counts = data[regressor].value_counts().sort_index()
    counts.plot(kind='bar', edgecolor='k', alpha=0.7, ax=ax1, color=colors)
    ax1.set_xlabel(regressors_names[i])
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{regressors_names[i]} Distribution and DR', fontweight='bold')

    # Calcular la TMO para cada categoría
    TMO = []
    categories = counts.index
    for category in categories:
        bin_data = data[data[regressor] == category]
        if len(bin_data) > 0:
            num_malos = bin_data['loan_status'].sum()  # Número de malos (asumiendo que 'loan_status' indica malos con 1 y buenos con 0)
            num_acreditados = len(bin_data)
            tmo = num_malos / num_acreditados  # Calcular TMO
        else:
            tmo = 0
        TMO.append(tmo)

    # Crear el segundo eje para la TMO
    ax2 = ax1.twinx()

    # Añadir la curva de TMO a la gráfica en el segundo eje
    ax2.plot(categories, TMO, color='red', marker='o', label='DR', linewidth=2)
    ax2.set_ylabel('Default Rate (DR)')

    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



###################################################################
#Categorical Distributions with Adjusted DR
###################################################################



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('credit_risk_dataset.csv')

# Eliminar filas con valores NaN en 'loan_int_rate'
data = data.dropna()
data['loan_intent'] = list(map(switch, data['loan_intent']))
data['person_home_ownership'] = list(map(switch2, data['person_home_ownership']))

# Lista de características categóricas
regressors_names = ['Home Ownership', 'Loan Intent', 'Loan Grade', 'Historical Default']
regressors = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Número total de buenos (préstamos no en mora)
total_buenos = len(data[data['loan_status'] == 0])

# Iterar sobre cada característica categórica para generar los gráficos
for i in range(len(regressors)):
    regressor = regressors[i]
    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots(figsize=( 3 * phi, 3))

    count = len(np.unique(data[regressor]))
    colors = [shap.plots.colors.red_blue((i) / (count - 1)) for i in range(count)]

    # Graficar el histograma de la característica categórica
    counts = data[regressor].value_counts().sort_index()
    counts.plot(kind='bar', edgecolor='k', alpha=0.7, ax=ax1, color = colors)
    ax1.set_xlabel(regressors_names[i])
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{regressors_names[i]} Distribution and Adjusted DR', fontweight='bold')


    # Calcular la TMO para cada categoría
    TMO = []
    ajuste = []
    categories = counts.index
    for category in categories:
        bin_data = data[data[regressor] == category]
        if len(bin_data) > 0:
            tmo = bin_data['loan_status'].mean()  # Proporción de mora en la categoría
            buenos_bin = len(bin_data[bin_data['loan_status'] == 0])
            ajuste.append(tmo / (buenos_bin / total_buenos))  # TMO ajustada
        else:
            tmo = 0
            ajuste.append(0)
        TMO.append(tmo)

    # Crear el segundo eje para la TMO y TMO ajustada
    ax2 = ax1.twinx()

    # Añadir la curva de TMO a la gráfica en el segundo eje
    #ax2.plot(categories, TMO, color='red', marker='o', label='TMO', linewidth=2)
    ax2.plot(categories, ajuste, color='blue', marker='o', label='Adjusted DR', linewidth=2)
    ax2.set_ylabel('Adjusted DR')

    # Añadir leyendas
    ax2.legend(loc='upper right')
    plt.grid(False)
    plt.tight_layout()
    plt.show()