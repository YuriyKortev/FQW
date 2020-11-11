import cv2 as cv
import urllib.request
import numpy as np
import random
import json
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from datetime import datetime

seconds = datetime.now()


def pxl2mm(x):
    """
    Перевод пикселей в мм
    :param x: Значение в пикселях
    :return: Значение мм
    """
    return x * 0.0242


def viewImage(image, name_of_window):
    """
    Просмотр изображения в окне
    :param image: Изображение
    :param name_of_window: Название окна
    """
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


string_to_class = {
    'crack': 1,
    'gap': 2
    #  'circle': 3
}


class_to_rgb = {
    1 : np.array([223,37,37], dtype='uint8'),
    2 : np.array([37,223,62], dtype='uint8')
}


def labels_to_rgb(labels):
    image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype='uint8')
    for key, val in class_to_rgb.items():
        image[labels==key]=val
        
    return image
    
    
def rgb_to_labels(rgb_image):
    labels = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    for key, val in class_to_rgb.items():
        labels[(rgb_image==val).all(-1)]=key
        
    return labels


def load_label(string_json, input_shape):
    """
    Функция парсит json с ссылками на разметки, загружает разметки, возвращает маску
    :param string_json: json с ссылками на разметки
    :param input_shape: Размерность изображения
    :return: Маска, где каждому пикселю соотвествует число 0, 1 или 2, в зависимости от класса
    """
    data = json.loads(string_json)
    objects = []

    if 'objects' in data:
        objects = data['objects']

    label = np.zeros(shape=(input_shape[0], input_shape[1]), dtype='uint8')
    for obj in objects:
        mask = get_image_by_url(obj['instanceURI'], input_shape, type='mask')
        if obj['value'] in string_to_class:
            label[mask == 255] = string_to_class[obj['value']]
    return label
    

def get_image_by_url(url, input_shape, type='mask'):
    """
    Загрузка изображения через url
    :param url: url к изображению
    :param input_shape: Размерность изображения
    :param type: Тип в зависимости от которого выбирается интерполяция для изменении размера изображения
    :return: Изображение заданного размера
    """
    with urllib.request.urlopen(url) as http:
        s = http.read()
        img_array = np.array(bytearray(s), dtype=np.uint8)
        img = cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
        if input_shape:
            inter = cv.INTER_NEAREST if type == 'mask' else cv.INTER_AREA
            img = cv.resize(img, (input_shape[1], input_shape[0]), interpolation=inter)

    return img


def load_image(url, input_shape):
    """
    Загрузка и форматирование изображения.
    :param url: Url изображения
    :param input_shape: Разрешение изображения
    :return: Изображение, которое может принять модель
    """
    img = get_image_by_url(url, input_shape, 'image')
    img.resize(img.shape[0], img.shape[1], 1)
    img = img.astype('float32')
    img[img == 0.0] = np.mean(img[img != 0.0])  # закрашивание черных областей
    img /= 255.  # нормализация

    return img


def load_image_by_path(path):
    img = cv.imread(path,0)
    #img[img == 0] = np.mean(img[img != 0])  # закрашивание черных областей
    
    return img
    
def preprocess_image(image):
    img = np.resize(image,(image.shape[0], image.shape[1], 1))
    img = img.astype('float32')
    img /= 255.0
    
    return img
    

def load_label_by_path(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    return img
    


class DataGenerator(Sequence):
    def __init__(self, paths, batch_size, input_shape, augmentations):
        self.paths = paths
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.paths)

    def __getitem__(self, idx):
        batch = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for sample in batch:
            image = load_image_by_path(sample[0])
            masks = load_label_by_path(sample[1])
            if self.augment:
                transformed = self.augment(image=image, mask=masks)
                labels = rgb_to_labels(transformed['mask'])
                batch_x.append(preprocess_image(transformed['image']))
                batch_y.append(tf.one_hot(labels, len(string_to_class) + 1, dtype='float32').numpy())
            else:
                batch_x.append(preprocess_image(image))
                labels = rgb_to_labels(masks)
                batch_y.append(tf.one_hot(labels, len(string_to_class) + 1, dtype='float32').numpy())
        return np.array(batch_x), np.array(batch_y)


def genData(path_x, path_y, batch_size, input_shape, augmentations=None):
    """
    Генератор, который каждый вызов возвращает пакет для обучения модели.
    :param augmentations: Последовательность преобразовний, которая применяется к изображениям и маскам
    :param input_shape: Форма входного тензора
    :param path_x: Путь к образцам
    :param path_y: Путь к маскам
    :param batch_size: Размер пакета
    :return: Возвращает кортеж из двух тензоров, 1-й - пакет с образцами, 2-й - пакет с масками
    """
    paths = list(zip(path_x.to_list(), path_y.to_list()))

    #random.seed(666)

    i = 0
    while True:
        image_batch = []
        mask_batch = []
        b=0
        while b < batch_size:
            if i == len(paths):
                i = 0
                random.shuffle(paths)
            sample = paths[i][0]
            label = paths[i][1]
            b += 1
            i += 1

            try:
                img = load_image(sample, input_shape)
                mask = load_label(label, input_shape)
            except:
                # global df
                # print('Cant load raw: {}'.format(df['Labeled Data'].to_list().index(sample)))
                b += -1
                continue

            if augmentations:
                transformed = augmentations(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask']

            image_batch.append(img)
            mask_batch.append(tf.one_hot(mask, len(string_to_class) + 1, dtype='float32').numpy())

        yield np.array(image_batch), np.array(mask_batch)


def drawWidth(thresh, image, type='crack'):
    """
    Функция принимает чернобелую маску с трещинами и исходное изображение,
    рисует обводку каждой трещины и ее максимальную ширину.
    :param type: Тип объекта: если не трещина, то рисовать ширину не надо
    :param thresh: Чернобелая маска с трещинами
    :param image: Исходное изображение
    :return: Массив с шириной каждой трещины
    """
    cx = image.shape[1] / thresh.shape[1]
    cy = image.shape[0] / thresh.shape[0]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        contour[:, :, 0] = contour[:, :, 0] * cx  # масштабирование контуров
        contour[:, :, 1] = contour[:, :, 1] * cy

        # print(cv.contourArea(contour))
        if cv.contourArea(contour) < 20000:
            continue

        cv.drawContours(image, [contour], -1, (0, 0, 255) if type is 'crack' else (0, 255, 0),
                        thickness=5)  # рисование границы трещины 20

        crack = np.zeros(image.shape)  # рисуется по одной трещине на чистом фоне и считается ширина
        cv.drawContours(crack, [contour], -1, 255, thickness=-1)
        
        if type is 'crack':

            max_width = 0
            point = (0, 0)

            #  считаю максимальную ширину, рисую линию и пишу текст
            for i, line in enumerate(crack):
                idxs = np.where(line == 255)
                if len(idxs) > 0 and len(idxs[0]) > max_width:
                    max_width = len(idxs[0])
                    point = (idxs[0][0], i)

            cv.line(image, point, (point[0] + max_width, point[1]), (0, 239, 250), 5) #20
            text = 'Cracks width: {} mm'.format(np.round(pxl2mm(max_width), 2))

            
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 2, 3)[0] #5 7

            point = (point[0] + int(max_width / 2) - int(text_size[0] / 2), point[1] - 35)

            if point[1] <= text_size[1]:
                point = (point[0], point[1] + text_size[1] + 70)

            if point[0] < 0:
                point = (0, point[1])
            elif point[0] + text_size[0] > image.shape[1]:
                point = (image.shape[1] - text_size[0], point[1])

            cv.putText(image, text, point, cv.FONT_HERSHEY_COMPLEX, 2, (0, 239, 250), 3) #5 7


def softmax_to_onehot(res):
    """
    Функция принимает выход модели с вероятностями и преобразует к маскам
    :param res: Выход модели
    :return: Маски каждого класса
    """
    classes = np.zeros(res.shape)
    am = res.argmax(axis=-1)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            classes[i][j][am[i][j]] = 1
    return classes
