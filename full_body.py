# Full body plate model with 17 output classes

# libraries
import pandas as pd
import importlib as imp
import random
import time
from myfuns import *
# imp.reload(myfuns)

random.seed = 1234


# keras imports
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.models import load_model

# paths
base_dir = '/Users/jgunnink/Documents/ktsa/'
data_dir = base_dir + 'data/'
aps_dir = base_dir + 'data/stage1_aps/'
# png_dir = data_dir + 'png/'

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

# xtab(zl, 'prob', 'zone_char').sort(['Lift'], ascending = 0)
xtab(zl, 'prob', 'zone_char')

# unique, counts = np.unique(rescaled, return_counts=True)

# get list of file names from os
file_names = [file for file in os.listdir(aps_dir) if file.endswith('.aps')] # all files
lab_names = [file + '.aps' for file in zl['id'].unique()] # labeled set
sub_names = [file for file in file_names if file not in lab_names] # submission set

# randomly sample cases for test and train sets
test_names = sorted(random.sample(lab_names, 117))
train_names = sorted([file for file in lab_names if file not in test_names])

# build training cases data
train_cases = np.empty(shape = (1030, 4096, 660), dtype='uint8')
gb(train_cases)

t0 = time.time()

for i, file in enumerate(train_names):
    if i%100 == 0:
        print(i, file)
    temp_plate = return_plate(read_data(aps_dir + file))
    rescaled_plate = (255.0 / temp_plate.max() * (temp_plate - temp_plate.min())).astype(np.uint8)
    train_cases[i,:,:] = rescaled_plate

t1 = time.time()
total = t1-t0
total

train_cases = train_cases.reshape(train_cases.shape[0], train_cases.shape[1], train_cases.shape[2], 1)

# build test cases data
test_cases = np.empty(shape = (117, 4096, 660), dtype='uint8')
gb(test_cases)

for i, file in enumerate(test_names):
    if i%100 == 0:
        print(i, file)
    temp_plate = return_plate(read_data(aps_dir + file))
    rescaled_plate = (255.0 / temp_plate.max() * (temp_plate - temp_plate.min())).astype(np.uint8)
    test_cases[i,:,:] = rescaled_plate

test_cases = test_cases.reshape(test_cases.shape[0], test_cases.shape[1], test_cases.shape[2], 1)

# vector of labels
train_filter = zl['file'].isin(train_names)
test_filter = zl['file'].isin(test_names)

train_df = zl[train_filter].sort_values(['id', 'zone_char'])
test_df = zl[test_filter].sort_values(['id', 'zone_char'])

train_labels = np.array(train_df[['prob']]).reshape(1030,17)
test_labels = np.array(test_df[['prob']]).reshape(117,17)

# test = train_cases[1,:,:,0]
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
m17.fit(x = train_cases, y = train_labels, validation_data = (test_cases, test_labels), batch_size=1, epochs = 1, verbose = 1)

model_file_name = 'zone13_model_8-3-2017.h5'

m17.save('m17_2017_08_28.h5')

# predict for unlabeled cases
sub_cases = np.empty(shape = (100, 4096, 660), dtype='uint8')
gb(sub_cases)

for i, file in enumerate(sub_names):
    if i%100 == 0:
        print(i, file)
    temp_plate = return_plate(read_data(aps_dir + file))
    rescaled_plate = (255.0 / temp_plate.max() * (temp_plate - temp_plate.min())).astype(np.uint8)
    sub_cases[i,:,:] = rescaled_plate

sub_cases = sub_cases.reshape(sub_cases.shape[0], sub_cases.shape[1], sub_cases.shape[2], 1)

preds = m17.predict(sub_cases, batch_size=1, verbose = 1)
pred_vec = preds.reshape(1700)

sub_labels = pd.read_csv(data_dir + 'sample_submission.csv')
sub_labels['id'] = sub_labels.Id.str[:32]
sub_labels['zone'] = sub_labels.Id.str[33:]
sub_labels['zone_num'] = sub_labels.zone.str.extract('(\d+)')
sub_labels['zone_char'] = sub_labels['zone_num'].str.zfill(2)
sub_labels = sub_labels.sort_values(['id', 'zone_char'])
sub_labels.head()
sub_labels['Probability'] = pred_vec
sl = sub_labels[['Id', 'Probability']]

sl.to_csv('m17_preds.csv', index=False)

outpreds = pd.concat([testing_ids.reset_index(drop=True),
                              pd.DataFrame(preds, columns=['prob'])], axis=1)
zone_predictions.to_csv('zone_' + str(zone_num) + '_predictions.csv', index=False)





