# fig = matplotlib.pyplot.figure(figsize=(16, 9))
# ax1 = fig.add_subplot(211)
# ax1.imshow(np.flipud(test[:, :].transpose()), cmap='viridis')
# ax2 = fig.add_subplot(212)
# ax2.imshow(np.flipud(rescaled[:, :].transpose()), cmap='viridis')

# write image to png
# im = Image.fromarray(rescaled)
# getsizeof(im)
# im.show()
# im.save(png_dir + "test.png")


# t0 = time.time()
# test = return_plate(data)
# t1 = time.time()
# total = t1-t0

# read in image data and put in a plate

samp = 'd7bba8e9f303107db7a0feab67d8e895.aps'
samp = '0043db5e8c819bffc15261b1f1ac5e42.aps'
data = read_data(aps_dir + samp)
plot_image(data[:,:,0])

test = data[:,:,0]
plot_image(test)

from scipy.misc import imresize
test2 = imresize(test, .25)

test2 = np.empty(shape = (16, 512, 660))
for i in np.arange(16):
    test2[i,:,:] = data[:,:,i]

plot_image(test2)

test2 = data.reshape(16, 512, 660)

plot_image(test2[15,:,:])
gb(test)
getsizeof(test)
getsizeof(test2)

#
# test = return_plate(data)
# plot_image(test)
#
# rescaled = (255.0 / test.max() * (test - test.min())).astype(np.uint8)
# plot_image(rescaled)
#
# getsizeof(test)
# getsizeof(rescaled)

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train.shape[1:]

from numpy import unique
np.unique(y_train)

test2 = train_cases[1029,:,:,0]
plot_image(test2)

# test = create_cases(train_names[:2], train_cases)
# plot_image(train_cases[32,:,:])

t0 = time.time()
t1 = time.time()
total = t1-t0
total


label_test = test_df['prob'].reshape(117,-1)
log_loss(label_test, pred_test)

# xtab(zl, 'prob', 'zone_char').sort(['Lift'], ascending = 0)
# xtab(zl, 'prob', 'zone_char')
# unique, counts = np.unique(rescaled, return_counts=True)

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape




