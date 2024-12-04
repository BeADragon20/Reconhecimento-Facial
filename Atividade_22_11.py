import cv2

# Adiciona o classificador
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#LÊ a imagem
img = cv2.imread('Familia3.jpg')

#Converte em escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecta as faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(10,10))

# Desenha ukm retangulo ao redor das faces 
for (x, y, l, a) in faces:
    cv2.rectangle(img, (x, y), (x+l, y+a), (255,0,0), 2)

# Mostra o resultado
cv2.imshow('Faces', img)
cv2.waitKey()
