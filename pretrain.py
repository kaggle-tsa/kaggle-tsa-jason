# libraries
import random
import pandas as pd
from myfuns import *
from scipy.misc import imresize
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support, log_loss, accuracy_score
from skimage import color
import matplotlib.pyplot as plt
# import importlib as imp
# imp.reload(myfuns)

# keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import load_model

# paths
base_dir = '/Users/jgunnink/Documents/ktsa/'
data_dir = base_dir + 'data/'
aps_dir = base_dir + 'data/stage1_aps/'

random.seed = 1234
pd.options.mode.chained_assignment = None

# magic numbers
resize_factor = 4
img_width = 512//resize_factor
img_height = 660//resize_factor
img_depth = 3
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

# read in sample submission file
sub_labels = pd.read_csv(data_dir + 'sample_submission.csv')
sub_labels['id'] = sub_labels.Id.str[:32]
sub_labels['zone'] = sub_labels.Id.str[33:]
sub_labels['zone_num'] = sub_labels.zone.str.extract('(\d+)', expand=False)
sub_labels['zone_char'] = sub_labels['zone_num'].str.zfill(2)
sub_labels = sub_labels.sort_values(['id', 'zone_char'])

# get list of file names from os
file_names = [file for file in os.listdir(aps_dir) if file.endswith('.aps')] # all files
lab_names = [file + '.aps' for file in zl['id'].unique()] # labeled set
sub_names = [file for file in file_names if file not in lab_names] # submission set

# randomly sample cases for test and train sets
test_names = sorted(random.sample(lab_names, num_test_cases))
train_names = sorted([file for file in lab_names if file not in test_names])

# function to create cases from data files
def create_cases(name_list, cases):
    for i, file in enumerate(name_list):
        if i % 100 == 0:
            print(i, file)
        case_data = read_data(aps_dir + file)
        case_temp = case_data[:, :, 0]
        case_temp = imresize(case_temp, 1/resize_factor)
        case_temp = ((255.0 / case_temp.max() * (case_temp - case_temp.min())).astype(np.uint8)) / 255
        case_temp = color.gray2rgb(case_temp)
        cases[i,:,:,:] = case_temp
    return cases

# build cases data for train, test, sub
train_cases = np.empty(shape = (num_train_cases, img_width, img_height, img_depth))
train_cases = create_cases(train_names, train_cases)

test_cases = np.empty(shape = (num_test_cases, img_width, img_height, img_depth))
test_cases = create_cases(test_names, test_cases)

sub_cases = np.empty(shape = (num_sub_cases, img_width, img_height, img_depth))
sub_cases = create_cases(sub_names, sub_cases)

# vector of labels
train_filter = zl['file'].isin(train_names)
test_filter = zl['file'].isin(test_names)

train_df = zl[train_filter].sort_values(['id', 'zone_char'])
test_df = zl[test_filter].sort_values(['id', 'zone_char'])

train_labels = np.array(train_df[['prob']]).reshape(num_train_cases, num_zones)
test_labels = np.array(test_df[['prob']]).reshape(num_test_cases, num_zones)

# pre-trained network
from keras.applications.inception_v3 import InceptionV3, preprocess_input

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # new FC layer, random init
predictions = Dense(17, activation='sigmoid')(x)
model = Model(input=base_model.input, output=predictions)

model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

model.fit(x = train_cases, y = train_labels, validation_data = (test_cases, test_labels), batch_size = 32, epochs = 5, verbose = 1)

model_save_name = 'inception_v3_10e'
model.save(model_save_name + '.h5')

test_pred = model.predict(test_cases, batch_size = num_test_cases, verbose = 1)
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

# predict for unlabeled cases and output submission file
sub_pred = model.predict(sub_cases, batch_size = num_sub_cases, verbose = 1)
sub_pred_vec = sub_pred.reshape(-1)
sl = sub_labels[['Id', 'Probability']]
sl['Probability'] = sub_pred_vec
sl.to_csv(model_save_name + '.csv', index=False)
