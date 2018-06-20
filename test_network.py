import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


def preprocess(input_img):
    #input_img_roi = input_img[53:,]
    #input_img_roi = input_img[80:130,10:310]
    #input_img_roi = input_img[64:134,60:260]
    input_img_roi = input_img[64:134,]
    #input_img_roi = input_img
    #print("image shape", input_img_roi.shape)
    output_img = cv2.cvtColor(input_img_roi, cv2.COLOR_RGB2GRAY)
    cv2.normalize(output_img, output_img, 0, 255, cv2.NORM_MINMAX)
    output_img = cv2.resize(output_img, (64, 64))
    return output_img

def augment(input_img, input_angle):
    
    output_images, output_angles = [],[]

    # Add center camera flipped images and angles to data set
    image_flipped = np.fliplr(input_img)
    angle_flipped = -input_angle
    output_images.append(image_flipped)
    output_angles.append(angle_flipped)

    # Add blurred image
    blurred = cv2.GaussianBlur(input_img, (5, 5), 0)
    output_images.append(blurred)
    output_angles.append(input_angle)
    
    # Add noisy image
    noise = np.zeros_like(input_img)
    cv2.randn(noise,(0),(45))
    noisy_img = input_img+noise                    
    output_images.append(noisy_img)
    output_angles.append(input_angle)

    return output_images, output_angles

def generator(samples, BATCH_SIZE=32, gen_type="train"):
    print("in gen")
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, BATCH_SIZE):
            batch_samples = lines[offset:offset+BATCH_SIZE]

            images = []
            angles = []
            #CORRECTION = 0.15 # too much to the right (drives on right line)
            #CORRECTION = 1.15 # a little to the right
            #CORRECTION = 2.15 # all over the place
            #CORRECTION = 1.75 # still too jittery
            #CORRECTION = 1.5   # still too jittery
            #CORRECTION = 0.75 # too jittery
            CORRECTION = 0.30
            CENTER = 0
            LEFT = 1
            RIGHT = 2

            for batch_sample in batch_samples:
                for cam in range(0,3):
                    # Add center camera images and angles to data set
                    img_name = '../data/IMG/'+batch_sample[cam].split('/')[-1]
                    img = cv2.imread(img_name)
                    img_gray_norm = preprocess(img)
                    if cam == CENTER:
                        img_angle = float(batch_sample[3])
                    elif cam == LEFT:
                        img_angle = float(batch_sample[3]) + CORRECTION
                    elif cam == RIGHT:
                        img_angle = float(batch_sample[3]) - CORRECTION
                    images.append(img_gray_norm)
                    angles.append(img_angle)
                    
                    # Augment
                    if gen_type == "train":
                        imgs_aug, angles_aug = augment(img_gray_norm, img_angle)
                        images += imgs_aug
                        angles += angles_aug

            # trim image to only see section with road
            X_train = np.array(images)
            X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
            y_train = np.array(angles)
            print("Total number of",gen_type,"samples in batch after data augmentation:", len(X_train)) 
            X_train, y_train = shuffle(X_train, y_train)
            yield X_train, y_train
 
if __name__ == '__main__':

    lines = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            #lines.append(line)
            temp_angle = float(line[3])
            if((temp_angle > -0.2 or temp_angle < 0.2) and np.random.random() < 0.50):
                continue
            lines.append(line)
    
    import sklearn
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    BATCH_SIZE = 32 
    train_generator = generator(train_samples, batch_size=BATCH_SIZE, gen_type="train")
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, gen_type="valid")

    print("Total number of training samples before data augmentation:", len(train_samples)) 
    print("Total number of validation samples before data augmentation:", len(validation_samples)) 
    
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Convolution2D, Activation, Cropping2D, BatchNormalization, Dropout
    from keras.utils.visualize_util import plot
    from keras.models import load_model
    
    model = Sequential()
    # Cropping speed up training by ~40 seconds
    # Cropping also helps the car get farther on the track
    # However, I decided not to use the Keras function for cropping
    # and perform it manually in drive.py instead

    # Adding in strides (subsamples) and increasing batch size 8->32 
    # sped up my training from ~1 hour to about ~1 minute
    
    model.add(Convolution2D(24,5,5,subsample=(2, 2), input_shape=(64,64,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5,subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(48,3,3,subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    
    # Added dense layers one by one (last to first)
    
    model.add(Dense(1164)) # Got to the bridge for the first time! But still fell in water :( 
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(100))  # Getting smoother, more right turns`
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1)) # Erratic driving
    model.compile(loss='mse', optimizer='adam')
    
    import matplotlib.pyplot as plt
    
    trained_model = model.fit_generator(
       train_generator, 
       samples_per_epoch = ((len(train_samples)*12)//BATCH_SIZE)*BATCH_SIZE,
       validation_data=validation_generator,
       nb_val_samples=len(validation_samples),#deleted *6 
       nb_epoch=5)

    model.save('model.h5')
    
    ### print the keys contained in the history object
    print(trained_model.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(trained_model.history['loss'])
    plt.plot(trained_model.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    
