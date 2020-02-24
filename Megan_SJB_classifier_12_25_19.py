import glob
import numpy as np
import spectral.io.envi as envi
import os
import scipy.io as sio
from datetime import datetime
start = datetime.now()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D, Convolution1D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
import os.path
import h5py
import time
import matplotlib
from matplotlib import pyplot as plt
from keras.layers.normalization import BatchNormalization
# from keras.utils import plot_model
import scipy as scipy
#from keras.utils import multi_gpu_model
from classification_label_to_image import classification_label2_image
import wandb
from wandb.keras import WandbCallback
#'WV_2m','WV_2m','RE_5m'
dimension = 5
deleted_channels=[]
labels=[]
x_train=[]
selected_sample_per_class=50000
balanced_option='balanced'
senario='atm_cor'
numChannels =8
epochs = 500
batchSize = 256
numOfClasses = 5
#'WV_2m','WV_5m','RE_5m'
locations = ['WV_5m']
for iter_locat in range(0, len(locations)):
    wandb.init(project='Megan_model_12_22',name=locations[iter_locat]+'Seperate_CNN_epoch_'+str(epochs)+'_dim_'+str(dimension),config={"hyper": "parameter"},reinit=True)

    def multi_gpu_cnn_model(numChannels):
        model = Sequential()
        if dimension == 1:
            # CNN for 5x5x8
            print("Designing CNN for input 1x1x8")
            model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(dimension, dimension, numChannels)))
            model.add(Dropout(0.01))
            # model.add(BatchNormalization(axis=-1))
            print(model.output)
            model.add(Convolution2D(16, (1, 1), activation='relu'))
            model.add(Dropout(0.01))
            # model.add(BatchNormalization(axis=-1))
            print(model.output)
            model.add(Flatten())
            model.add(Dense(numOfClasses, activation='softmax'))
            print(model.output)
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if dimension == 3:
            # CNN for 5x5x8
            print("Designing CNN for input 3x3x8")
            model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(dimension, dimension, numChannels)))
            model.add(Dropout(0.01))
            # model.add(BatchNormalization(axis=-1))
            print(model.output)
            model.add(Convolution2D(16, (3, 3), activation='relu'))
            model.add(Dropout(0.01))
            # model.add(BatchNormalization(axis=-1))
            print(model.output)
            model.add(Flatten())
            model.add(Dense(numOfClasses, activation='softmax'))
            print(model.output)
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if dimension == 5:
            # CNN for 5x5x8
            print("Designing CNN for input 5x5x8")
            model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(dimension, dimension, numChannels)))
            model.add(Dropout(0.01))
            print(model.output)
            model.add(Convolution2D(16, (4, 4), activation='relu'))
            model.add(Dropout(0.01))
            print(model.output)
            model.add(Flatten())
            model.add(Dense(numOfClasses, activation='softmax'))
            print(model.output)
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    import sys
    orig_stdout = sys.stdout
    multi_model = multi_gpu_cnn_model(numChannels)

    patch_crop_point=int(np.floor(dimension/2))
    save_directory = './seperate_model_12_22/'+locations[iter_locat]+'CNN_del_chl_'+str(deleted_channels)+'_epochs_'+str(epochs)+'_dim_'+str(dimension)+'/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print('directory created', save_directory)

    cnnFileName = 'Single_classifier_'+balanced_option+'_dimension_' + str(
        dimension) + "_channels_" + str(
        numChannels) + "_epochs_" + str(epochs) + "_batchSize_" + str(batchSize) + "numOfClasses" + str(
        numOfClasses) + ".h5"
    model = Sequential()
    if os.path.isfile(save_directory+cnnFileName):
        model = Sequential()
        multi_model = load_model(save_directory+cnnFileName)
        multi_model.summary()
    else:
        class_name = ["Deep_Water", "Sand", "Seagrass", "Land", "Intertidal"]
        print(class_name)
        for class_numb in range(0, len(class_name)):
            location = locations[iter_locat]
            training_data_directory=location+"/ROIs/"
            A = (glob.glob(training_data_directory + class_name[class_numb] + "*.hdr"))
            print(A)
            sample_data = []
            for patchnumb in range(0, len(A)):
                # lib = envi.open('WV320180331162748M00_1_atmCorr.hdr','WV320180331162748M00_1_atmCorr.til')
                lib = envi.open(A[patchnumb], A[patchnumb][0:-4])
                print(lib)
                im = lib
                a_zeros = np.zeros([lib.shape[0], lib.shape[1]])
                if dimension>1:
                    a_zeros[0:, 0:patch_crop_point] = 1
                    a_zeros[0:patch_crop_point, 0:] = 1
                    a_zeros[-patch_crop_point:, 0:] = 1
                    a_zeros[0:, -patch_crop_point:] = 1
                if locations[iter_locat]=='RE_5m':
                    nan_values_location = np.where((lib[:, :, [0]] < 0) & (lib[:, :, 1] < 0) & (lib[:, :, [2]] < 0) &
                                                   (lib[:, :, [3]] < 0) & (lib[:, :, [4]] < 0))
                else:
                    nan_values_location = np.where((lib[:, :, [0]] < 0) & (lib[:, :, 1] < 0) & (lib[:, :, [2]] < 0) & (lib[:, :, [3]] < 0) & (
                                                    lib[:, :, [4]] < 0) & (lib[:, :, [5]] < 0) & (lib[:, :, [6]] < 0) & (lib[:, :, [7]] < 0))
                for i in range(len(nan_values_location[0])):
                    a_zeros[nan_values_location[0][i], nan_values_location[1][i]] = 1

                indeex_loc = np.where(a_zeros == 0)

                rows_loc = indeex_loc[0]
                colms_loc = indeex_loc[1]
                print('rows_loc colms_loc', lib.shape[0], lib.shape[1], colms_loc.shape[0], rows_loc.shape[0])
                data_divided_into = 1
                length_all_location = int(rows_loc.shape[0])
                division_len = int(np.ceil(length_all_location / data_divided_into));
                count_data_division = 0
                for iteration in range(0, length_all_location, division_len):
                    print('code running')
                    count_data_division = count_data_division + 1
                    print('division number', count_data_division, '    :.....')
                    print('division number', length_all_location, '    :.....', )
                    if count_data_division == data_divided_into:
                        data_iter_end = length_all_location;
                    else:
                        data_iter_end = iteration + division_len;

                    data_length = (data_iter_end - iteration);
                    f = np.zeros([data_length, dimension, dimension, lib.shape[2]]);
                    image_index = np.zeros([data_length, 2]);

                    for data_iter in range(iteration, data_iter_end):
                        # print(data_iter)
                        l = rows_loc[data_iter];
                        m = colms_loc[data_iter];

                        e = np.zeros([dimension, dimension,  lib.shape[2]]);
                        e[0:dimension, 0:dimension,0: lib.shape[2]] = im[l -patch_crop_point:l+patch_crop_point+1, m -patch_crop_point:m+patch_crop_point+1, :];
                        image_index[(data_iter - iteration), :] = [l, m];

                        f[(data_iter - iteration), :, :, :] = e;


                    class data_structs:
                        pass
                    images_multi = data_structs()
                    images_multi.data = f
                    images_multi.image_index = image_index
                sample_data.append(f)
            sample_data = np.concatenate(sample_data)

            print('Sample data Shape', sample_data.shape)
            if balanced_option == 'balanced':

                if sample_data.shape[0] > selected_sample_per_class:
                    index_downsample = np.random.choice(sample_data.shape[0], selected_sample_per_class, replace=False)
                elif sample_data.shape[0] < selected_sample_per_class:
                    index_downsample = np.random.choice(sample_data.shape[0], selected_sample_per_class)
                sample_data = sample_data[index_downsample, :, :, :]
                print('after blancing Sample data Shape', sample_data.shape)
                # downsample_label = downsample_label[index_downsample]
                labels.append(class_numb * np.ones(selected_sample_per_class))
                x_train.append(sample_data)
                del sample_data
            elif balanced_option == 'unbalanced':
                labels.append(class_numb * np.ones(int(sample_data.shape[0])))
                x_train.append(sample_data)
                del sample_data

        x_train = np.concatenate(x_train)
        labels = np.concatenate(labels);x_train = np.delete(x_train, deleted_channels, 3)
        print(x_train.shape, labels.shape)
        y_train = np_utils.to_categorical(labels)

        f = open(save_directory +'command_window_'+'.txt', 'w')
        sys.stdout = f
        history=multi_model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize,validation_split=0.1, shuffle=True,callbacks=[WandbCallback()])
        print(history)
        sys.stdout = orig_stdout
        print(history.history.keys())
        print(history)
        #  "Accuracy"
        plt.clf()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(save_directory+'acc.png')
        plt.clf()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(save_directory+'loss.png')
        multi_model.save(save_directory+cnnFileName)
        plt.clf()
        #
        multi_model.summary()
    #testing_data_subdir=['WV320180331162746M00_1_AtmCorr','WV320180331162747M00_1_AtmCorr','WV320180331162748M00_1_AtmCorr']
    #testing_datas=['WV320180331162746M000_1_AtmCorr','WV320180331162747M00_1_AtmCorr','WV320180331162748M000_1_AtmCorr']
    #locations = ['25Sep19','28Nov19']
    #   for iter_locat in range(0, len(locations)):
    location = locations[iter_locat]
    method_name = location + '.mat'
    testing_data_directory = location+"/Image/"
    A = (glob.glob(testing_data_directory + "*.hdr"))
    #lib = envi.open('./SJB_25Sep19/'+filename+'.hdr','./SJB_25Sep19/'+filename)
    lib = envi.open(A[0], A[0][0:-4]+'.til')
    im=lib
    a_zeros=np.zeros([lib.shape[0],lib.shape[1]])
    if dimension>1:
        a_zeros[0:, 0:patch_crop_point] = 1
        a_zeros[0:patch_crop_point, 0:] = 1
        a_zeros[-patch_crop_point:, 0:] = 1
        a_zeros[0:, -patch_crop_point:] = 1
    rows_half = lib.shape[0];  # columns=11134; rows=4403;
    colms_half = lib.shape[1];
    if locations[iter_locat] == 'RE_5m':
        nan_values_location = np.where((lib[:, :, [0]] < 0) & (lib[:, :, 1] < 0) & (lib[:, :, [2]] < 0) &
                                       (lib[:, :, [3]] < 0) & (lib[:, :, [4]] < 0))
    else:
        nan_values_location = np.where(
            (lib[:, :, [0]] < 0) & (lib[:, :, 1] < 0) & (lib[:, :, [2]] < 0) & (lib[:, :, [3]] < 0) & (
                    lib[:, :, [4]] < 0) & (lib[:, :, [5]] < 0) & (lib[:, :, [6]] < 0) & (lib[:, :, [7]] < 0))
    for i in range(len(nan_values_location[0])):
        a_zeros[nan_values_location[0][i], nan_values_location[1][i]] = 1
    testing_result_class1 = np.zeros([rows_half, colms_half]);
    print('nan_values_location',nan_values_location[0].shape)
    print(testing_result_class1.shape)
    # 3591, 3591
    indeex_loc = np.where(a_zeros == 0)
    rows_loc=indeex_loc[0]
    colms_loc=indeex_loc[1]
    print(lib,rows_loc.shape,colms_loc.shape,a_zeros.shape)
    data_divided_into=100
    length_all_location=int(rows_loc.shape[0])
    division_len = int(np.ceil(length_all_location / data_divided_into));
    print('division_len',division_len)
    count_data_division = 0
    for iteration in range(0, length_all_location,division_len):
        print('code running')
        count_data_division = count_data_division + 1
        print('division number',count_data_division,'    :.....')
        if count_data_division == data_divided_into:
            data_iter_end = length_all_location;
        else:
            data_iter_end = iteration + division_len;

        data_length = (data_iter_end - iteration) ;
        f = np.zeros([data_length,dimension, dimension,  lib.shape[2]]);
        image_index = np.zeros([ data_length,2]);

        for data_iter in range (iteration,data_iter_end):
            #print(data_iter)
            l = rows_loc[data_iter];
            m = colms_loc[data_iter];
            e = np.zeros([dimension, dimension, lib.shape[2]]);
            e[0:dimension, 0:dimension,0: lib.shape[2]] = im[l -patch_crop_point:l + patch_crop_point+1, m - patch_crop_point:m + patch_crop_point+1,0 : lib.shape[2]];
            image_index[ (data_iter - iteration),:]=[l, m];
           # print(image_index.shape)
            f[(data_iter - iteration),:,:,: ]=e;
            #print(f.shape)
        class data_structs:
             pass
        images_multi=data_structs()
        images_multi.data=f
        images_multi.image_index=image_index;f = np.delete(f, deleted_channels, 3)
        #sio.savemat(save_directory + str(count_data_division) + '_spidercab_bay_atm_cor_part1_testing ' + '.mat', {'images_multi': images_multi})

        predicted_label = multi_model.predict(f, batch_size=1024)
        print(predicted_label.shape)
        y_pred_arg = np.argmax(predicted_label, axis=1);
        print(y_pred_arg.shape)
        # sio.savemat('test_filename_' + f

        test_location = image_index
        predict_label_len = len(y_pred_arg)
        for index_predict in range(0, predict_label_len):
            testing_result_class1[int(test_location[ index_predict,0]), int(test_location[ index_predict,1])] =  y_pred_arg[index_predict]+1 ;
    testing_result_class1 = np.uint8(testing_result_class1)
    sio.savemat(save_directory + location + '_results_' + method_name, {'testing_result_class1': testing_result_class1})
    #sio.savemat(save_directory + location + '_results_2_' + method_name, {'testing_result_class2': testing_result_class2})
    difference = datetime.now() - start
    print(difference)
    im3 = classification_label2_image(testing_result_class1,
                                      save_directory + location + '_' + method_name[0:-4])

    difference = datetime.now() - start
    print(difference)
    difference = datetime.now() - start
    print(difference);





