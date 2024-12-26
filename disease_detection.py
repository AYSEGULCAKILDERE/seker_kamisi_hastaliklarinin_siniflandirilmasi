import tensorflow as tf

# Veri setlerini oluşturma
training_set = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
)

test_set = tf.keras.utils.image_dataset_from_directory(
    'data/test',
    image_size=(64, 64),
    batch_size=32
)

# Veri ön işleme
# Piksel değerlerini 0-1 arasına normalize etme
normalization_layer = tf.keras.layers.Rescaling(1./255)
training_set = training_set.map(lambda x, y: (normalization_layer(x), y))
test_set = test_set.map(lambda x, y: (normalization_layer(x), y))

# Veri artırma (data augmentation) için
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Veri artırmayı training set'e uygulama
training_set = training_set.map(lambda x, y: (data_augmentation(x, training=True), y))

# Model tanımlama
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation="relu"),
    tf.keras.layers.Dense(units=5, activation="softmax")
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme ve history nesnesini kaydetme
history = cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Modeli kaydetme
cnn.save('model.keras')

# Eğitim sürecini görselleştirme
import matplotlib.pyplot as plt

# Grafik boyutunu ayarlama
plt.figure(figsize=(12, 4))

# Kayıp (Loss) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

# Doğruluk (Accuracy) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()
