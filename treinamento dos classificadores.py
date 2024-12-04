import cv2
import os #Recursos do Sistema Operacional
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create(threshold = 500)
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos',f) for f in os.listdir('fotos')] #cria uma lista com o caminho de cada uma das fotos na pasta "fotos"
    faces = []
    ids = []
    for caminhoImagem in caminhos:

        
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) #Faz a leitura da imagem converte em escala de cinza e grava na variável
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = getImagemComId()

print('Treinando...')
eigenface.train(faces, ids) #Realiza o treinamento do classificador
eigenface.write('classificadorEigen.yml') #Grava na pasta o arquivo com o clssificador gerado

fisherface.train(faces, ids) #Realiza o treinamento do classificador
fisherface.write('classificadorFisher.yml') #Grava na pasta o arquivo com o clssificador gerado

lbph.train(faces, ids) #Realiza o treinamento do classificador
lbph.write('classificadorLBPH.yml') #Grava na pasta o arquivo com o clssificador gerado

print('Treinamento realizado')

