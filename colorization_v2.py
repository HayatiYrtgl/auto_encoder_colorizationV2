from up_down_functions import create_model
from keras.callbacks import ModelCheckpoint
import numpy as np

# load data
x_gray = np.load("../DATASET/land/gray_x.npy")
x_color = np.load("../DATASET/land/color_x.npy")

print(x_color.shape, x_gray.shape)

x_train_gray = x_gray[:5500]
x_train_color = x_color[:5500]

x_test_color = x_color[5500:]
x_test_gray = x_gray[5500:]

model = create_model()
model.compile(loss="mae", metrics=["accuracy"], optimizer="adam")

cp = ModelCheckpoint(filepath="../models/colorizer_v3", monitor="accuracy")

model.fit(x_train_gray, x_train_color, validation_data=(x_test_gray, x_test_color), epochs=50, batch_size=16,
          callbacks=[cp])
