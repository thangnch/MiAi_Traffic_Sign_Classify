import os
import pandas as pd
#from scipy.misc import imread
from imageio import imread
import math
import numpy as np
import cv2
import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

def load_data(input_size = (64,64), data_path =  'GTSRB/Final_Training/Images'):

    pixels = []
    labels = []
    # Loop qua các thư mục trong thư mục Images
    for dir in os.listdir(data_path):
        if dir == '.DS_Store':
            continue

        # Đọc file csv để lấy thông tin về ảnh
        class_dir = os.path.join(data_path, dir)
        info_file = pd.read_csv(os.path.join(class_dir, "GT-" + dir + '.csv'), sep=';')

        # Lăp trong file
        for row in info_file.iterrows():
            # Đọc ảnh
            pixel = imread(os.path.join(class_dir, row[1].Filename))
            # Trích phần ROI theo thông tin trong file csv
            pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
            # Resize về kích cỡ chuẩn
            img = cv2.resize(pixel, input_size)

            # Thêm vào list dữ liệu
            pixels.append(img)

            # Thêm nhãn cho ảnh
            labels.append(row[1].ClassId)

    return pixels, labels

# Đường dẫn ảnh
data_path = 'GTSRB/Final_Training/Images'
pixels, labels = load_data(data_path=data_path)


def split_train_val_test_data(pixels, labels):

    # Chuẩn hoá dữ liệu pixels và labels
    pixels = np.array(pixels)
    labels = keras.utils.np_utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)
    X = pixels[randomize]
    print("X=", X.shape)
    y = labels[randomize]

    # Chia dữ liệu theo tỷ lệ 60% train và 40% còn lại cho val và test
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    val_size = int(X_val.shape[0] * 0.5) # 50% của phần 40% bên trên
    X_val, X_test = X_val[:val_size], X_val[val_size:]
    y_val, y_test = y_val[:val_size], y_val[val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_data(pixels, labels)


def build_model(input_shape=(64,64,3), filter_size = (3,3), pool_size = (2, 2), output_size = 43):
    model = Sequential([
        Conv2D(16, filter_size, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(16, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, filter_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),
        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.summary()
    return model

# Build model với kích thước đầu vào 64x64 và output là 43 classes
model = build_model(input_shape=(64,64,3), output_size=43)

# Train model
epochs = 10
batch_size = 16

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                               validation_data=(X_val, y_val))

model.save("traffic_sign_model.h5")
#model = keras.models.load_model("traffic_sign_model.h5")

# Kiểm tra model với dữ liệu mới
print(model.evaluate(X_test, y_test))