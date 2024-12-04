import cv2

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create() #Cria o reconhecedor
reconhecedor.read("classificadorEigen.yml") #Utiliza o arquivo com o treinamento realizado
largura = 220
altura = 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL #Fonte para imprimir o nome da  pessoa da face reconhecida
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30,30))
    
    for (x,y,l,a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y+a, x:x+l],(largura, altura))
        cv2.rectangle(imagem, (x, y), (x+l, y+a), (0,0,255), 2)
        id, confianca = reconhecedor.predict(imagemFace) #Realiza o reconhecimento Facial
        if id ==11:
            nome = "Elisangela"
        elif id == 18:
            nome = "Gustavo"
        else:
            nome = "Desconhecido"
        cv2.putText(imagem, nome, (x,y+(a+30)), font, 2, (0,0,255)) #Adiciona o texto com o id da pessoa detectada
        cv2.putText(imagem, str(confianca), (x, y+(a+50)), font, 1, (0,0,255))
    
    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()