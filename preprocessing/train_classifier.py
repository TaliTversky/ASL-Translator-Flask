import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

max_length = max(len(seq) for seq in data_dict['data'])
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


# data_YOU = []
# labels_YOU = []

# # Iterate over each pair of data and label
# for data, label in zip(data_dict['data'], data_dict['labels']):
#     if label in ['Y', 'O', 'U']:
#         data_YOU.append(data)  # Append data to data_YOU list
#         labels_YOU.append(label)  # Append label to labels_YOU list

# # Create a dictionary with data and corresponding labels
# data_with_labels_YOU = {'data': data_YOU, 'labels': labels_YOU}

# f = open('dataYOU.pickle', 'wb')
# pickle.dump({'data': data_YOU, 'labels': labels_YOU}, f)

# data_with_labels_YOU contains data and corresponding labels for records marked with labels 'Y', 'O', and 'U'