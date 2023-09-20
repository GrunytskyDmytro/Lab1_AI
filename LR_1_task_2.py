from sklearn import preprocessing

# Надання позначок вхідних даних
Input_labels = [2.3, -1.6, 6.1, -2.4, -1.2, 4.3, 3.2, 5.5, -6.1, -4.4, 1.4, -1.2]

# Бінарізація
binarizer = preprocessing.Binarizer(threshold=2.1)
input_data_binarized = binarizer.transform([Input_labels])
print("Binarized data:\n", input_data_binarized)

# Виключення середнього
input_data_mean_removed = preprocessing.scale(Input_labels)
print("\nMean removed data:\n", input_data_mean_removed)

# Масштабування
min_max_scaler = preprocessing.MinMaxScaler()
input_data_scaled = min_max_scaler.fit_transform([Input_labels])
print("\nScaled data:\n", input_data_scaled)

# Нормалізація
input_data_normalized_l1 = preprocessing.normalize([Input_labels], norm='l1')
input_data_normalized_l2 = preprocessing.normalize([Input_labels], norm='l2')
print("\nNormalized data l1:\n", input_data_normalized_l1)
print("\nNormalized data l2:\n", input_data_normalized_l2)
