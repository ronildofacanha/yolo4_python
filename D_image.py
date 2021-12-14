import cv2
import numpy as np  # Tabalha com a parte cientifica (Vetores e Matrizes)
import time  # tempo de execução

# ARQUIVOS
input_file = 'arquivos/persons.jpeg'
weights_path = 'arquivos/yolov4.weights'
cfg_path = 'arquivos/yolov4.cfg'
names_path = 'arquivos/coco.names'

threshold = 0.8  # Nivel de confiança?
threshold_NMS = 0.3
boxes = []
confidences = []
classIDs = []

# Carregar nome das classes
LABELS = []
with open(names_path, 'r') as names:
    LABELS = [cname.strip() for cname in names.readlines()]
# print(len(LABELS))

# SETANDO OS PARAMETROS DA REDE NEURAL
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# CAMADAS DE SAIDA E ENTRADA DO YOLO
#print('CAMADAS DE SAIDA: ', net.getUnconnectedOutLayers())
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1]
               for i in net.getUnconnectedOutLayers()]  # SAIDA DE LAYER NAME
#print("SAIDA ", len(layer_names))

# Gerar cores aleatorias
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


def showImage(img):
    cv2.imshow('img', img)
    # print(img.shape) # FORMATO DA IMAGEM
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread(input_file)
(H, W) = image.shape[:2]
# H, W = image.shape[:2]  # Altura e largura da imagem
# CALCULAR PROCESSAMENTO
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(layer_names)
end = time.time()
tempo = (end - start)
# print(len(layerOutputs))
print("[INFO] YOLO took {:.2f} seconds".format(end - start))

# NIVEL DO CLASSIFICADOR E CONFIGURAÇÃO DAS VARIAVEIS
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # NIVEL DE CONFIANÇA
        if confidence > threshold:
            box = detection[0:4] * np.array([W, H, W, H])  # CAIXA
            (centerX, centerY, width, height) = box.astype('int')
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

#print("CAIXAS: ", len(boxes))
#print("CONFIANCA: ", len(confidences))


def f_imagem(image, i, confidences, boxes, COLORS, LABELS, mostrar_texto=True):
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])

    # extract the bounding box coordinates
    color = [int(c) for c in COLORS[classIDs[i]]]

    fundo = np.full((image.shape), (0, 0, 0), dtype=np.uint8)

    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

    if(mostrar_texto):
        print('> ' + text)
        print(x, y, w, h)

    cv2.putText(fundo, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    fx, fy, fw, fh = cv2.boundingRect(fundo[:, :, 2])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, -1)
    cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), color, 3)
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)
    return image, x, y, w, h


idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold_NMS)

if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():

        if LABELS[classIDs[i]] == 'person':
            image, x, y, w, h = f_imagem(
                image, i, confidences, boxes, COLORS, LABELS, False)

showImage(image)
