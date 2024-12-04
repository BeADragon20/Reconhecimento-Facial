import cv2

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0) #Indica a camera a ser utilizada

amostra = 1 #Controla quantas fotos foram capturadas

numeroAmostras = 25 #limite max de fotos

id = input('Digite o seu n° da chamada: ')

largura = 220 #Tamanho padrao para imagens
altura = 220 #Tamanho padrao para imagens

print('Capturando as Faces...')

while (True):
    conectado, imagem = camera.read() # Faz a leitura do fraime da webcam
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converter em escala de Cinza
    #Detecta as Faces
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor= 1.5, minSize=(150,150))

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0,0,255), 2) #Desenha o retangulo ao redor da face
        if cv2.waitKey(1) == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y+a, x:x+a], (largura, altura)) #Redimenciona a imagem
            cv2.imwrite("fotos/aluno."+str(id)+"."+str(amostra)+".jpg",imagemFace)
            print("[foto "+str(amostra)+ " capturada com sucesso]")
            amostra += 1

    cv2.imshow('Face', imagem) #exibe a imagem
    k = cv2.waitKey(1) #Aguarda, por 1 milisegundo, uma tecla ser precissionada
    if k==27:
        break #Fecha a tela se a tecla ESC for pressionada
    
    if (amostra > numeroAmostras):
        break
camera.release() #Libera a memória
cv2.destroyAllWindows() # Fecha todas as janelas

