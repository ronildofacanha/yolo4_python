# CARREGAR AS DEPENDÊNCIAS
from typing import Text
import cv2
import time
import numpy as np
import time
import time
import concurrent.futures
from multiprocessing.pool import ThreadPool
# Contador

# ARQUIVOS
input_file = 'arquivos/carros1.mp4'
weights_path = 'arquivos/yolov4-tiny.weights'
cfg_path = 'arquivos/yolov4-tiny.cfg'
names_path = 'arquivos/coco.names'


def foo(tempo, baz):
    acabou = False

    while tempo:
        print(str(tempo) + '\n')
        time.sleep(1)
        tempo -= 1
    if(tempo == 0):
        print('TEMPO FINALIZDO')
        acabou = True
    return acabou


def tempoDeterminado(frame, obj):

    if obj == "person":
        _tempo = 10
        Text = 'TEMPO PARA PESSOA: '
        cv2.putText(frame, Text+str(_tempo)+'s', (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 3)

    elif obj == "chair":
        _tempo = 15
        Text = 'TEMPO PARA CADEIRANTE: '
        cv2.putText(frame, Text+str(_tempo)+'s', (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 3)


# CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# CARREGA AS CLASSES
LABELS = []
with open(names_path, 'r') as names:
    LABELS = [cname.strip() for cname in names.readlines()]

cap = cv2.VideoCapture(0)

# code da main Abaixo

# CARREGANDO OS PESOS DA REDE NEURAL
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


def capturar():
    # LENDO OS FRAMES DO VIDEO
    while True:

        # CAPTURA DO FRAME
        _, frame = cap.read()

        # COMEÇO DA CONTAGEM DOS MS
        start = time.time()

        # DETECÇÃO
        classes, scores, boxes = model.detect(frame, 0.5, 0.5)

        # FIM DA CONTAGEM DOS MS
        end = time.time()

        # PERCORRER AS DETECÇÕES
        for(classid, score, box) in zip(classes, scores, boxes):

            # GERANDO UMA COR PARA CLASSE
            color = COLORS[int(classid) % len(COLORS)]
            # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
            # label = f"{LABELS[classid]} : {score}"
            label = LABELS[classid] + '{: .2f}'.format(score)
            # DESENHANDO A BOX DA DETECÇÃO
            cv2.rectangle(frame, box, color, 2)
            # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            tempoDeterminado(frame, LABELS[classid])

            '''
            if LABELS[classid] == 'person':
                # GERANDO UMA COR PARA CLASSE
                color = COLORS[int(classid) % len(COLORS)]
                # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
                # label = f"{LABELS[classid]} : {score}"
                label = 'Pedestre + {: .2f}'.format(score)
            # DESENHANDO A BOX DA DETECÇÃO
                cv2.rectangle(frame, box, color, 2)
                # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
                cv2.putText(frame, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                '''
        # MOSTRANDO A IMAGEM
        # ESPERA DA RESPOSTA
        cv2.putText(frame, 'QUANTIDADE DE DETECOES: '+str(len(boxes)), (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == 27:
            break


pool = ThreadPool(processes=1)
async_result = pool.apply_async(foo, (3, 'foo'))  # tuple of args for foo
return_val = async_result.get()

if(return_val):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(capturar)


# LIBERAÇÃO DA CAMERA E DESTROI TODAS AS JANELAS
cap.release()
cv2.destroyAllWindows()
