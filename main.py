
#%%
import numpy as np
import os
import pandas as pd
import tlsh 
import random

np.random.seed(42)

def get_entry(f, label):
    hash_value = tlsh.hash(f)
    hash.fromTlshStr(hash_value)
    dataset_entry = {'q1ratio': hash.q1ratio,
                'q2ratio': hash.q2ratio,
                'lvalue': hash.lvalue,
                'checksum': hash.checksum(0),
                'malware': label}
    for bi in range(0,128):
        dataset_entry['bucket ' + str(bi)] = hash.bucket_value(bi)  
    return dataset_entry
#%%
benign_arm = []
benign_bytes = []
malware_arm = []
malware_bytes = []                
                
root=os.getcwd()
abs = root
dirs=os.listdir(root)
for dir in dirs:
    if (dir == 'data'):
        os.chdir(abs + '/data/benign/arm')
        for file in os.listdir(os.getcwd()):
            hash = tlsh.Tlsh()
            with open(file, 'rb') as f:
                f_b = f.read()
                benign_bytes.append(f_b)
                benign_arm.append(get_entry(f_b, 0))
        os.chdir(abs + '/data/benign/mips')
        for file in os.listdir(os.getcwd()):
            with open(file, 'rb') as f:
                f_b = f.read()
                benign_bytes.append(f_b)
        os.chdir(abs + '/data/malware/arm')
        for file in os.listdir(os.getcwd()):
            hash = tlsh.Tlsh()
            with open(file, 'rb') as f:
                f_b = f.read()
                malware_bytes.append(f_b)
                malware_arm.append(get_entry(f_b, 1))
        os.chdir(abs + '/data/malware/mips')
        for file in os.listdir(os.getcwd()):
            with open(file, 'rb') as f:
                f_b = f.read()
                malware_bytes.append(f_b)
#%%
def injection(binary, label):
    if label == 0:
        i = random.randint(0, len(malware_bytes) - 1)
        return binary + malware_bytes[i][:-1*len(malware_bytes[i])//1000]
    else:
        i = random.randint(0, len(benign_bytes) - 1)
        return binary + benign_bytes[i][:-1*len(benign_bytes[i])//1000]
#%%        
dataset_benign = pd.DataFrame.from_dict(benign_arm)
dataset_malware = pd.DataFrame.from_dict(malware_arm)
# %%
os.chdir(abs)
dataset_benign.to_csv("benign_arm.csv")
dataset_malware.to_csv("malware_arm.csv")
# %%
from sklearn.model_selection import train_test_split
dataset = pd.concat([dataset_benign, dataset_malware])
X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'malware'], dataset['malware'], test_size = 0.2, shuffle = True)
# %%
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
clf_1 = RandomForestClassifier(n_estimators = 10, max_depth=6) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf_1.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf_1.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix     

ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', ax=ax)
# %%
# coverages
paths, indices = clf_1.decision_path(X_train)
# %%
# nodes of a single indicator
# print(paths.toarray()[0]) 
# %%
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import vstack
misclassfied = 0
false_positive = 0
false_negative = 0
mis_order = []
new_states = 0
accuracy = []
epochs = 500
# choose a seed randomly
for i in range(0, epochs):
    if random.random() < .5:
        new_i = random.randint(0, len(benign_bytes) - 1)
        candidate = benign_bytes[new_i]
        label = 0
        new_seed = injection(candidate, label)
    else:
        new_i = random.randint(0, len(malware_bytes) - 1)
        candidate = malware_bytes[new_i]
        label = 1
        new_seed = injection(candidate, label)

    new_seed_dict = get_entry(new_seed, label)
    new_seed_dict.pop('malware')
    new_seed_ready = pd.Series(new_seed_dict)
    #check if there is misclassification
    pred = clf_1.predict(np.reshape(new_seed_ready.values, (1, -1)))
    if pred != label:
        misclassfied+=1   
        mis_order.append(1)
        if pred == 1:
            false_positive+=1
        else:
            false_negative+=1
    else:
        mis_order.append(0)
    #calculate the distances    
    distances = []
    new_path, ind = clf_1.decision_path(np.reshape(new_seed_ready.values, (1, -1)))
    for path in paths:
        distance = pairwise_distances(path.toarray(), new_path.toarray(), metric='manhattan')
        distances.append(distance)
    if np.min(distances) > 90:
        X_train.append(new_seed_ready.rename('new'))
        y_train.append(pd.Series(label).rename('new'))
        clf_1.fit(X_train, y_train)
        paths, indices = clf_1.decision_path(X_train)
        y_pred = clf_1.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        if label == 0:
            benign_bytes.append(new_seed)
        else:
            malware_bytes.append(new_seed)
        new_states += 1
# %%
y_pred_1 = clf_1.predict(X_train)
y_pred_2 = clf_1.predict(X_test)
print("ACCURACY OF THE MODEL (TRAIN): ", metrics.accuracy_score(y_train, y_pred_1))
print("ACCURACY OF THE MODEL (TEST): ", metrics.accuracy_score(y_test, y_pred_2))
print(misclassfied)
print(accuracy)
print(mis_order)
print(false_positive)
print(false_negative)
#%%
np.add.reduceat(mis_order, np.arange(0, len(mis_order), 10))