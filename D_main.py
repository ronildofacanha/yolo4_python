import cv2  # OpenCV
import numpy as np  # Tabalha com a parte cientifica (Vetores e Matrizes)
import time
import dlib
import time
import threading
import math
tempoGB = 0
statusSemaforo = 'null'
extent = False
count = 0
semaforoAberto = True


def thread_delay(null):
    global tempoGB
    global count
    global semaforoAberto
    count = 0
    while count < tempoGB:
        print("TEMPO:->:", count)
        time.sleep(1)
        count += 1

    if count == tempoGB:
        tempoGB = 0
        if semaforoAberto == False:
            semaforoAberto = True
        else:
            semaforoAberto = False

        print('SEMAFORO', '->', semaforoAberto)
    return 0


def Calculating_time_extent(classes):

    global tempoGB
    global extent
    global semaforoAberto

    if tempoGB == 0 and tempoGB < 20:
        tempoGB = 20
        extent = False
        t1 = threading.Thread(target=thread_delay, args=(0,))
        t1.start()
        print("TEMPO: PESSOA ", tempoGB)
        # Text = 'TEMPO PARA PESSOA: '
        # cv2.putText(frame, Text+str(tempoGB)+'s', (0, 25),
        #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 3)
    if classes == "chair" and extent == False and tempoGB < 30:
        tempoGB += 10
        print("TEMPO: CADEIRANTE ", tempoGB)
        extent = True
        # cv2.putText(frame, Text+str(tempoGB)+'s', (0, 25),
        #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 3)


def imageShow(img):
    cv2.imshow('TELA DE CAPTURA', img)
    # print(img.shape) # FORMATO DA IMAGEM
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# CONSTUÇÃO DO BLOB
def blobImage(net, img, ln):
    start = time.time()
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_Outputs = net.forward(ln)
    end = time.time()
    # print("[TEMPO DE DETECÇÃO] {:.2f} seconds".format(end - start))
    return net, img, layer_Outputs


# REALIZAR DETECÇÃO
def detectionImage(_detection, _threshold, _AllBoxes, _AllConfidences, _AllClassesID, _img):
    (H, W) = _img.shape[:2]
    scores = _detection[5:]
    classeID = np.argmax(scores)
    confidence = scores[classeID]
    if confidence > _threshold:
        box = _detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype('int')
        x = int(centerX - (width/2))
        y = int(centerY - (height/2))
        _AllBoxes.append([x, y, int(width), int(height)])
        _AllConfidences.append(float(confidence))
        _AllClassesID.append(classeID)

    return _AllBoxes, _AllConfidences, _AllClassesID


# TAMANHO DO VIDEO
def reSizeX(_width, _height, _widthMax=600):
    if(_width > _widthMax):
        newSize = _width / _height
        video_width = _widthMax
        video_height = int(video_width/newSize)
    else:
        video_width = _width
        video_height = _height
    return video_width, video_height

# CRIAR CAIXAS


def createBoxes(_img, i, _confidences, _boxes, _COLORS, _LABELS, _AllClassesID):
    (x, y) = (_boxes[i][0], _boxes[i][1])
    (w, h) = (_boxes[i][2], _boxes[i][3])
    color = [int(c) for c in _COLORS[_AllClassesID[i]]]
    text = "{}: {:.4f}".format(
        _LABELS[_AllClassesID[i]], _confidences[i])

    fundo = np.full((_img.shape), (0, 0, 0), dtype=np.uint8)

    cv2.putText(fundo, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    fx, fy, fw, fh = cv2.boundingRect(fundo[:, :, 2])
    cv2.rectangle(_img, (x, y), (x + w, y + h), color, 2)

    cv2.rectangle(_img, (fx, fy), (fx+fw, fy+fh), color, -1)
    cv2.rectangle(_img, (fx, fy), (fx+fw, fy+fh), color, 3)
    cv2.putText(_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)

    return _img, x, y, w, h


# CAPTURA FRAME A FRAME
def DetectionX():
    # ARQUIVOS
    input_file = 'arquivos/lv1.mp4'
    weights_path = 'arquivos/yolov4-tiny.weights'
    cfg_path = 'arquivos/yolov4-tiny.cfg'
    names_path = 'arquivos/coco.names'
    # CONFIG PRECISÃO
    threshold = 0.3  # Nivel de confiança?
    threshold_NMS = 0.4
    font_smal, font_big = 0.4, 0.6
    font_tipe = cv2.FONT_HERSHEY_SIMPLEX
    fontLine = 2  # inteiro
    amostrar_exibir = 20
    amostra_atual = 0
    # CARREGANDO NOME DAS CLASSES
    with open(names_path, 'r') as names:
        LABELS = [cname.strip() for cname in names.readlines()]
    # CARREGANDO ARQUIVOS
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # CORES DAS CLASSES
    np.random.seed(50)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')
    # CAMADAS DE SAIDA
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
    # CARREGAR VIDEO
    cap = cv2.VideoCapture(0)
    connected, video = cap.read()
    video_height = video.shape[0]
    video_width = video.shape[1]

    # CLIQUE [1]  EXIT
    video_width, video_height = reSizeX(video_width, video_height)

    while(cv2.waitKey(1) < 0):
        _connected, _frame = cap.read()
        if not _connected and type(_frame) == type(None):
            print("VIDEO NULL")
            break
        t = time.time()
        _frame = cv2.resize(_frame, (video_width, video_height))
        try:
            (H, W) = _frame.shape[:2]
        except:
            print('VIDEO SHAPE ERRO')
            continue

        net, _frame, layerOutputs = blobImage(net, _frame, ln)
        AllBoxes = []
        AllConfidences = []
        AllClassesID = []

        for output in layerOutputs:
            for detection in output:
                AllBoxes, AllConfidences, AllClassesID = detectionImage(
                    detection, threshold, AllBoxes, AllConfidences, AllClassesID, _frame)

        objects = cv2.dnn.NMSBoxes(AllBoxes, AllConfidences,
                                   threshold, threshold_NMS)

        if len(objects) > 0:
            for i in objects.flatten():

                Calculating_time_extent(LABELS[AllClassesID[i]])

                if semaforoAberto:
                    _frame, x, y, w, h = createBoxes(
                        _frame, i, AllConfidences, AllBoxes, COLORS, LABELS, AllClassesID)

        cv2.putText(_frame, "PROCSSAMENTO {:.2f}s".format(time.time()-t),
                    (20, video_height-20), font_tipe, font_big, (255, 255, 255), fontLine, lineType=cv2.LINE_AA)

        cv2.putText(_frame, "TEMPO: "+str(count)+'s', (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 3)
        cv2.putText(_frame, "SEMAFORO: "+str(semaforoAberto), (0, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # t1 = threading.Thread(target=thread_delay, args=(5,))
        # t1.start()
        # print(tempoGB)
        if semaforoAberto:
            semaforoImage = cv2.imread("arquivos/sTrue.png")
            cv2.putText(_frame, "N DETECCOES: "+str(len(AllBoxes)), (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        else:
            semaforoImage = cv2.imread("arquivos/sFalse.png")
        cv2.imshow('SEMAFORO', semaforoImage)

        imageShow(_frame)


def main():
    print("MAIN!")
    DetectionX()
    # t1 = threading.Thread(target=thread_delay, args=(5,))
    t2 = threading.Thread(target=DetectionX, args=('',))
    # t1.start()
    t2.start()
    print("terminou: ")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
