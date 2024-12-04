import cv2

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0) #Indica a camera a ser utilizada

while (True):
    conectado, imagem = camera.read() # Faz a leitura do fraime da webcam
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converter em escala de Cinza
    #Detecta as Faces
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor= 1.5, minSize=(100,100))

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0,0,255), 2) #Desenha o retangulo ao redor da face

    cv2.imshow('Face', imagem) #exibe a imagem
    k = cv2.waitKey(1) #Aguarda, por 1 milisegundo, uma tecla ser precissionada
    if k==27:
        break #Fecha a tela se a tecla ESC for pressionada

camera.release() #Libera a memória
cv2.destroyAllWindows() # Fecha todas as janelas
