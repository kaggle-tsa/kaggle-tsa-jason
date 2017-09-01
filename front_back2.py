# libraries
import time
import random
import pandas as pd
from myfuns import *
from scipy.misc import imresize
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support, log_loss, accuracy_score
import matplotlib.pyplot as plt
# import importlib as imp
# imp.reload(myfuns)

# keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model

# paths
base_dir = '/Users/jgunnink/Documents/ktsa/'
data_dir = base_dir + 'data/'
aps_dir = base_dir + 'data/stage1_aps/'

random.seed = 1234

# magic numbers
img_width = 256
img_height = 165
num_train_cases = 1030
num_test_cases = 117
num_sub_cases = 100
num_zones = 17


# process labels file
zone_labels = pd.read_csv(data_dir + 'stage1_labels.csv')
zone_labels['id'] = zone_labels.Id.str[:32]
zone_labels['zone'] = zone_labels.Id.str[33:]
zone_labels['zone_num'] = zone_labels.zone.str.extract('(\d+)')
zone_labels['zone_char'] = zone_labels['zone_num'].str.zfill(2)
zone_labels.head()
zl = zone_labels[['id', 'zone_char', 'zone_num', 'Probability']]
zl.columns = ['id', 'zone_char', 'zone_num', 'prob']
zl['file'] = zl['id'] + '.aps'

# get list of file names from os
file_names = [file for file in os.listdir(aps_dir) if file.endswith('.aps')] # all files
lab_names = [file + '.aps' for file in zl['id'].unique()] # labeled set
sub_names = [file for file in file_names if file not in lab_names] # submission set

# randomly sample cases for test and train sets
test_names = sorted(random.sample(lab_names, 117))
train_names = sorted([file for file in lab_names if file not in test_names])

# function to create cases from data files
def create_cases(name_list, cases):
    for i, file in enumerate(name_list):
        if i % 100 == 0:
            print(i, file)
        case_data = read_data(aps_dir + file)
        case_temp = case_data[:, :, 0]
        case_temp = np.row_stack((case_temp, case_data[:, :, 8]))
        case_temp = imresize(case_temp, .25)
        case_temp = ((255.0 / case_temp.max() * (case_temp - case_temp.min())).astype(np.uint8)) / 255
        cases[i,:,:] = case_temp
    return cases

# build cases data for train, test, sub
train_cases = np.empty(shape = (num_train_cases, img_width, img_height))
train_cases = create_cases(train_names, train_cases)
train_cases = train_cases.reshape(train_cases.shape[0], train_cases.shape[1], train_cases.shape[2], 1)

test_cases = np.empty(shape = (num_test_cases, img_width, img_height))
test_cases = create_cases(test_names, test_cases)
test_cases = test_cases.reshape(test_cases.shape[0], test_cases.shape[1], test_cases.shape[2], 1)

sub_cases = np.empty(shape = (num_sub_cases, img_width, img_height))
sub_cases = create_cases(sub_names, sub_cases)
sub_cases = sub_cases.reshape(sub_cases.shape[0], sub_cases.shape[1], sub_cases.shape[2], 1)

# vector of labels
train_filter = zl['file'].isin(train_names)
test_filter = zl['file'].isin(test_names)

train_df = zl[train_filter].sort_values(['id', 'zone_char'])
test_df = zl[test_filter].sort_values(['id', 'zone_char'])

train_labels = np.array(train_df[['prob']]).reshape(num_train_cases, num_zones)
test_labels = np.array(test_df[['prob']]).reshape(num_test_cases, num_zones)

# test = test_cases[116,:,:,0]
# plot_image(test)

# specify network
m17 = Sequential()
m17.add(Convolution2D(32, (3, 3), activation ='elu', input_shape = train_cases.shape[1:]))
m17.add(Dropout(0.10))
m17.add(MaxPooling2D(pool_size = (2, 2)))
m17.add(Convolution2D(32, (3, 3), activation ='elu'))
m17.add(Dropout(0.10))
m17.add(MaxPooling2D(pool_size = (2, 2)))
m17.add(Flatten())
m17.add(Dense(17, activation='sigmoid'))

# compile
m17.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
m17.summary()

# fit
m17.fit(x = train_cases, y = train_labels, validation_data = (test_cases, test_labels), batch_size = 32, epochs = 10, verbose = 1)

m17.save('m17_fb_25_10.h5')
# m17 = load_model('xxx')

# data augmentation
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# datagen.fit(train_cases)
#
# m17.fit_generator(datagen.flow(train_cases, train_labels, batch_size = 20), steps_per_epoch=len(train_cases) / 20, epochs = 10)


# test set performance metrics
test_pred = m17.predict(test_cases, batch_size = 117, verbose = 1)
test_pred_vec = test_pred.reshape(-1)
test_label_vec = test_labels.reshape(-1)

plotROC(test_label_vec, test_pred_vec)

threshold = .5
test_pred_thresh = np.where(test_pred_vec > threshold, 1, 0)

ll = log_loss(test_label_vec, test_pred_vec)
acc = accuracy_score(test_label_vec, test_pred_thresh)
area_under_roc = roc_auc_score(test_label_vec, test_pred_vec)
precision, recall, fscore, support = precision_recall_fscore_support(test_label_vec, test_pred_thresh, average='binary')

print('Log Loss: ' + str(ll))
print('AUC: ' + str(area_under_roc))
print('Accuracy: ' + str(acc))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('FScore: ' + str(fscore))


# predict for unlabeled cases
sub_pred = m17.predict(sub_cases, batch_size = 100, verbose = 1)
sub_pred_vec = sub_pred.reshape(-1)

sub_labels = pd.read_csv(data_dir + 'sample_submission.csv')
sub_labels['id'] = sub_labels.Id.str[:32]
sub_labels['zone'] = sub_labels.Id.str[33:]
sub_labels['zone_num'] = sub_labels.zone.str.extract('(\d+)')
sub_labels['zone_char'] = sub_labels['zone_num'].str.zfill(2)
sub_labels = sub_labels.sort_values(['id', 'zone_char'])
sub_labels.head()
sub_labels['Probability'] = sub_pred_vec
sl = sub_labels[['Id', 'Probability']]

sl.to_csv('m17_fb_25_1.csv', index=False)






