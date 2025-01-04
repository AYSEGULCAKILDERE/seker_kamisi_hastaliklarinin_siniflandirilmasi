import cv2
import tensorflow as tf
import numpy as np

# Modeli yükleme
model = tf.keras.models.load_model('model.keras')

# Sınıfların isimlerini tanımlama
class_names = ['Healthy', 'mosaic', 'RedRot', 'Rust', 'Yellow']

# Kamera açma
cap = cv2.VideoCapture(0)

# Kameradan görüntü alıp model ile sınıflandırma
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Görüntüyü 64x64 boyutlarına küçültme
    resized_frame = cv2.resize(frame, (64, 64))
    
    # Görüntüyü normalleştirme (0-1 arası)
    normalized_frame = resized_frame / 255.0
    
    # Görüntüyü modelin beklediği formatta hazırlama
    input_frame = np.expand_dims(normalized_frame, axis=0)
    
    # Modeli kullanarak tahmin yapma
    predictions = model.predict(input_frame)
    
    # Tahmini sınıf
    predicted_class = np.argmax(predictions)
    
    # Sonucu ekrana yazdırma
    label = class_names[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    text = f'{label}: {confidence:.2f}%'
    
    # Ekranda tahmin sonucunu ve sınıfı gösterme
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Kamera Görüntüsü", frame)
    
    # 'q' tuşuna basıldığında çıkış yapma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
