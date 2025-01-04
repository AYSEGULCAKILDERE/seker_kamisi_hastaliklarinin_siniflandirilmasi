import tensorflow as tf

# Veri setlerini oluşturma
training_set = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(64, 64),  # Orijinal boyut
    batch_size=32,
    shuffle=True
)

test_set = tf.keras.utils.image_dataset_from_directory(
    'data/test',
    image_size=(64, 64),
    batch_size=32
)

# Veri ön işleme
normalization_layer = tf.keras.layers.Rescaling(1./255)
training_set = training_set.map(lambda x, y: (normalization_layer(x), y))
test_set = test_set.map(lambda x, y: (normalization_layer(x), y))

# Basit veri artırma
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)
])

# Model mimarisi
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2),
    
    tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    
    tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation="softmax")
])

# Model derleme
cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model eğitimi
history = cnn.fit(
    training_set,
    validation_data=test_set,
    epochs=50
)

# Modeli kaydet
cnn.save('model.keras')

# Eğitim sürecini görselleştirme
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()
