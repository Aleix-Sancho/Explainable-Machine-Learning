def get_SHAP_values(model, data, data_x, x0):
  #@model: A keras model
  #@data: An array of observations from the input space from where \phi_0 is estimated
  #@data_x: An array of observations from the input space whose SHAP Values will be computed
  #@x_0: A reference observation with the expected value of each feature.
  import numpy as np
  from math import factorial as fact

  def get_binary_subsets(n):#Gets every binary array of length n
    secuencias = [format(i, f'0{n}b') for i in range(2 ** n)]
    secuencias_np = [np.array(list(map(int, list(seq)))) for seq in secuencias]
    return secuencias_np

  def weight(S):#Computes the weight from the Shapley Values Formula given the coallition S
    return fact(np.sum(S)) * fact(n - np.sum(S) - 1) / fact(n)

  def get_shap(i):#Computes the SHAP Value for the player i
    one = np.ones(n)

    subsetsF_diff_i = [cadena for cadena in subsetsF if cadena[i] == 0]#Every subset of F that does not contain i

    #SHAP Values Formula
    shap_i = 0
    for S in subsetsF_diff_i:
      S_plus_i = np.copy(S)
      S_plus_i[i] = 1

      if(np.all(S == 0)):
        f_S = expected_value
      else:
        f_S = model.predict([np.multiply(S, x) + np.multiply(one - S, [x0])], verbose = 0)[0][0]
        #np.multiply(S, x) + np.multiply(one - S, x0) is equivalent to E(f|x_S) where f is linearly aproximated

      f_S_plus_i = model.predict([np.multiply(S_plus_i, x) + np.multiply(one - S_plus_i, [x0])], verbose = 0)

      shap_i = shap_i + weight(S) * (f_S_plus_i - f_S)
    return(shap_i[0][0])

  expected_value = np.mean(model.predict(data, verbose = 0))
  n = model.input_shape[1] #number of features

  subsetsF = get_binary_subsets(n)#Gets every subset of F

  shap_values = []
  for x in data_x:
    shap_values_x = list(map(get_shap, list(range(n))))
    shap_values.append(shap_values_x)

  return(shap_values)


def get_Kernel_SHAP_values(model, data, data_x, x0):
  import numpy as np
  from scipy.special import comb
  from sklearn.linear_model import LinearRegression

  def model_expected_function(model, x, x0, S):
    one = np.ones(len(S))
    x_S = list(np.multiply(S, x) + np.multiply(one - S, x0))
    return model.predict([x_S], verbose=0)[0][0]

  def get_binary_subsets(n):#Gets every binary array of length n
    secuencias = [format(i, f'0{n}b') for i in range(2 ** n)]
    secuencias_np = [np.array(list(map(int, list(seq)))) for seq in secuencias]
    return secuencias_np

  n = model.input_shape[1] #number of features

  expected_value = np.mean(model.predict(data, verbose = 0))

  subsetsF = get_binary_subsets(n)[1:-1] #every subset excep np.zero and np.one as they get an infinite weight

  kernel_shap_values = []
  for x in data_x:
    x = list(x)
    X_kernel = []
    Y_kernel = []
    W = []
    f_x = model.predict([x], verbose = 0)[0][0]

    for S in subsetsF:#Kernel SHAP Sample
      Y_kernel.append(model_expected_function(model, x, x0, S) - f_x * S[0] + expected_value * (S[0] - 1))

      X_obs = []
      for i in list(range(n))[1:]:
        X_obs.append(S[i] - S[0])
      X_kernel.append(X_obs)

      n_z = sum(S)
      W.append((n - 1 ) / (comb(n, n_z) * n_z * (n - n_z)))

    modelo = LinearRegression(fit_intercept=False)
    modelo.fit(X_kernel, Y_kernel, sample_weight = W)

    kernel_shap_values_x = np.insert(np.copy(modelo.coef_), 0, f_x - expected_value - sum(modelo.coef_))
    kernel_shap_values.append(list(kernel_shap_values_x))

  return kernel_shap_values


def get_Deep_SHAP_Values(model, data_x, x0):
  import numpy as np
  def get_neurons_output(x):
    i_x = x
    neurons_output = [np.array(i_x)]
    for i in list(range(n_layers)):
      i_layer = model.layers[i] #current layer
      i_bias = i_layer.weights[1] #shape = n_inputs x 1
      i_weights = i_layer.weights[0] #shape = n_neurons x n_inputs

      #computes the neurons output
      i_linear_output = i_bias + np.dot(np.transpose(i_weights),i_x)
      neurons_output.append(np.array(i_linear_output))

      i_output = i_layer.activation(i_linear_output)
      i_x = np.copy(i_output)

      neurons_output.append(np.array(i_x))
    return neurons_output #neurons_output.length == 2*n_layers, as for every layer it saves the linear and non linear outputs

  n_layers = len(model.layers) #does not count the input layer
  contribution_scores = []

  #FORWARD PASS
  neurons_output_x0 = get_neurons_output(x0)
  for x in data_x:
    neurons_output_x = get_neurons_output(x)

    diff_from_ref = []
    for array1, array2 in zip(neurons_output_x, neurons_output_x0):
      resta = array1 - array2
      diff_from_ref.append(resta)

    #BACKPROPAGATION PASS
    i = len(diff_from_ref) - 1 #last layer after activation function

    #Initializing
    C_in_out = diff_from_ref[i]
    m_in_out = [1.]

    while(i >= 1):
      if(i % 2 == 0):#activation iteration
          m_in_out = (diff_from_ref[i] / diff_from_ref[i - 1]) * np.transpose(m_in_out)#Chain Rule for Multipliers, one input
          #C_in_out does not change in this step

      else:#linear iteration
          i_layer_weigths = model.layers[int(i/2)].weights[0]#weights of the current layer

          m_in_out = np.dot(i_layer_weigths, np.transpose(m_in_out))#CRfM
          C_in_out = diff_from_ref[i-1] * np.transpose(m_in_out)#Multiplier def.
      i = i - 1

    contribution_scores.append(list(C_in_out))
  return contribution_scores

def SHAP_plot(data, shap_values, title, feature_names = []):
  import matplotlib.pyplot as plt
  import numpy as np
  import shap

  observations = data
  shap_values = shap_values

  np.random.seed(42)
  num_samples = observations.shape[0]
  num_features = observations.shape[1]
  np.random.normal(0, 0.1, 50)

  order = []
  for i in range(data.shape[1]):
    max = np.max(np.array(shap_values)[:,i])
    min = abs(np.min(np.array(shap_values)[:,i]))
    order.append(np.max([max, min]))
  order = np.argsort(order)

  if(feature_names == []):
    feature_names = [f'Feature {i}' for i in order]
  else:
    aux = []
    for i in range(data.shape[1]):
      aux.append(feature_names[order[i]])
    feature_names = aux

  fig, ax = plt.subplots(figsize=(8,10))

  spacing = .5  
  y_positions = np.arange(num_features) * spacing

  j = 0
  for i in order:
      shap_values_i = np.array(shap_values)[:,i]
      observation_i = list(observations[:,i])

      ax.scatter( shap_values_i,  [y_positions[j]] * num_samples + np.random.normal(0, 0.035, num_samples),
                c=observation_i, cmap=shap.plots.colors.red_blue, vmin=data.min(), vmax=data.max(), alpha=0.8, s=20)
      j = j + 1

  ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha = 0.2)
  spacing = .5  
  ax.set_yticks(y_positions)
  ax.set_yticklabels(feature_names)
  ax.set_xlabel(title)
  ax.set_ylim(-spacing*3/4, (num_features - 1/4) * spacing) 
  ax.set_title("")

  cbar = plt.colorbar(ax.collections[0])
  cbar.set_ticks([data.min(), data.max()])
  cbar.set_ticklabels(["Low", "High"])
  cbar.set_label('Feature Value', labelpad=-20)


  plt.grid(False)
  plt.tight_layout()
  plt.show()