import os
import numpy as np
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
from sys import getsizeof

# read data
def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype='S1', count=20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype='S1', count=80))
    h['energy_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['config_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['file_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['trans_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['scan_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['data_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype='S1', count=16))
    h['frequency'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['mat_velocity'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['num_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['spare00'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['band_width'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['spare01'] = np.fromfile(fid, dtype=np.int16, count=5)
    h['polarization_type'] = np.fromfile(fid, dtype=np.int16, count=4)
    h['record_header_size'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['word_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['word_precision'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['min_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['max_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['avg_data_value'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['data_scale_factor'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['data_units'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['surf_removal'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['edge_weighting'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['x_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['y_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['z_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['t_units'] = np.fromfile(fid, dtype=np.uint16, count=1)
    h['spare02'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['x_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_return_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['scan_orientation'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['scan_direction'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['data_storage_order'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['scanner_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['x_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['t_inc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['num_x_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
    h['num_y_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
    h['num_z_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
    h['num_t_pts'] = np.fromfile(fid, dtype=np.int32, count=1)
    h['x_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_speed'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['x_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_acc'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['x_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_motor_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['x_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_encoder_res'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype='S1', count=8))
    h['depth_recon'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['x_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_max_travel'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['adc_type'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['spare06'] = np.fromfile(fid, dtype=np.int16, count=1)
    h['scanner_radius'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['x_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['y_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['z_offset'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['t_delay'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['range_gate_start'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['range_gate_end'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['ahis_software_version'] = np.fromfile(fid, dtype=np.float32, count=1)
    h['spare_end'] = np.fromfile(fid, dtype=np.float32, count=10)
    return h

def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512)  # skip header
    if extension == '.aps' or extension == '.a3daps':
        if (h['word_type'] == 7):  # float32
            data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
        elif (h['word_type'] == 4):  # uint16
            data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
        data = data * h['data_scale_factor']  # scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy()  # make N-d image
    elif extension == '.a3d':
        if (h['word_type'] == 7):  # float32
            data = np.fromfile(fid, dtype=np.float32, count=nx * ny * nt)
        elif (h['word_type'] == 4):  # uint16
            data = np.fromfile(fid, dtype=np.uint16, count=nx * ny * nt)
        data = data * h['data_scale_factor']  # scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy()  # make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype=np.float32, count=2 * nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0, :, :, :].copy()
        imag = data[1, :, :, :].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag

# plot image
def plot_image(data, i=0):
    fig = matplotlib.pyplot.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    if len(data.shape) == 3:
        ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
    else:
        ax.imshow(np.flipud(data[:, :].transpose()), cmap='viridis')

# crosstab functions
def freq(x, byvar):
    freq = x.groupby(byvar).agg([np.size])[x.columns[0]]
    freq.columns = ['Num_Obs']
    freq['Tot_Obs'] = freq['Num_Obs'].sum()
    freq['Pct_Obs'] = freq['Num_Obs']/freq['Tot_Obs']
    freq = freq.reset_index()
    return(freq)

def xtab(x, pos, byvar):
    xtab = x.groupby(byvar).agg([np.sum, np.size])[pos]
    xtab.columns = ['Num_Pos', 'Num_Obs']
    xtab['Pos_Rate'] = xtab['Num_Pos']/xtab['Num_Obs']
    xtab['Tot_Pos'] = xtab['Num_Pos'].sum()
    xtab['Tot_Obs'] = xtab['Num_Obs'].sum()
    xtab['Pct_Obs'] = xtab['Num_Obs']/xtab['Tot_Obs']
    xtab['Lift'] = xtab['Pos_Rate']/(xtab['Tot_Pos']/xtab['Tot_Obs'])
    xtab = xtab.reset_index()
    return(xtab)

def return_plate(img):
    img_plate = img[:,:,0]
    for i in range(2, 16, 2):
    # for i in range(1, 16, 1):
        img_plate = np.row_stack((img_plate, img[:,:,i]))
    return img_plate

def gb(x):
    return print(str(round(getsizeof(x)/(1024*1024*1024),3)) + ' GB')

def plotROC(labels, probs):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probs)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

