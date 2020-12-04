# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:54:16 2020

@author: JuanManuel
"""
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches
import os
import glob
from sklearn import svm
from skimage.transform import radon
# Modelos a comparar
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Métricas a utilizar
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z']

def KNN(X_train, X_test, y_train, y_test):
    #Obteniendo el mejor valor de K a partir del conjunto de validación con mejor accuracy
    scores = []
    maxscore = [0,0] # [k,score]
    for k in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric='euclidean', metric_params=None,algorithm='brute')
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)
        if score > maxscore[1]:
            maxscore = [k,score]
        
    plt.figure()
    plt.xlabel('Parametro k')
    plt.ylabel('Accuracy')
    plt.scatter(range(1,40), scores)
    
    print('El mejor resultado fue:',maxscore)
    knn = KNeighborsClassifier(n_neighbors = maxscore[0],weights='distance',metric='euclidean', metric_params=None,algorithm='brute')
    knn.fit(X_train, y_train)
    
    pred = knn.predict(X_test)
    # Matriz de confusion
    print(confusion_matrix(y_test, pred))
    # Reporte de clasificación
    print(classification_report(y_test, pred))
    # Matthews
    print('Matthews: ',matthews_corrcoef(y_test, pred))
    return knn

def SVM(X_train, X_test, y_train, y_test):
    kernels=['linear', 'poly', 'rbf', 'sigmoid']
    max_score = 0
    max_kernel = 500
    max_degree = 500
    for Kernel in range(4):
        if Kernel == 1:
            for Degree in range(1,11):
                msv = svm.SVC(kernel=kernels[Kernel],degree=Degree)
                msv.fit(X_train, y_train)
                score = msv.score(X_test, y_test)
                if score > max_score:
                    max_score = score
                    max_kernel = Kernel
                    max_degree = Degree
        else:
            msv = svm.SVC(kernel=kernels[Kernel])
            msv.fit(X_train, y_train)
            score = msv.score(X_test, y_test)
            if score > max_score:
                    max_score = score
                    max_kernel = Kernel
    
    print('El mejor Kernel es', kernels[max_kernel],'de grado',max_degree,'con un accuracy de:',max_score)
    msv = svm.SVC(kernel=kernels[max_kernel])
    msv.fit(X_train, y_train)
    print('Accuracy of SVM classifier on test set: {:.2f}'
         .format(msv.score(X_test, y_test)))
    return msv

def NN(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(40,40,30,15,8),activation='relu',random_state=1, max_iter=1000,verbose=False)
    clf.fit(X_train, y_train)
    print('Accuracy por Red Neuronal con sklearn en conjunto de entrenamiento: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy por Red Neuronal con sklearn en conjunto de validación: {:.2f}'.format(clf.score(X_test, y_test)))
    # Se generan las métricas
    pred = clf.predict(X_test)
    # Matriz de confusion
    print(confusion_matrix(y_test, pred))
    # Reporte de clasificación
    print(classification_report(y_test, pred))
    # Matthews
    print('Matthews: ',matthews_corrcoef(y_test, pred))
    return clf

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        for each in range(10):
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            binary_image = img_details < threshold_otsu(img_details)
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
    
    abc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
                    'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
    nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
    for letra in range(len(target_data)):
        for x in range(len(abc)):
            if target_data[letra] == abc[x]:
                target_data[letra] = nums[x]
    return (np.array(image_data), np.array(target_data))

def Imp_Placa(knn,msv,clf):
    print('\nPredicciones:')
    print('---------------------')
    print('KNN\t   |SVM\t   |NN')
    for x in range(len(knn)):
        print(knn[x],end='')
    print('\t',end='|')
    for x in range(len(knn)):
        print(msv[x],end='')
    print('\t',end='|')   
    for x in range(len(knn)):
        print(clf[x],end='')
    print('\n---------------------')
    
# MAIN
print('Reading data')
#Xtotal, ytotal = Base_Datos()
training_dataset_dir = './training_images'
Xtotal, ytotal = read_training_data(training_dataset_dir)
print('Reading data completed')

# Se generan los conjuntos de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(Xtotal, ytotal,random_state = 1)
scaler= MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNN(X_train, X_test, y_train, y_test)
msv = SVM(X_train, X_test, y_train, y_test)
clf = NN(X_train, X_test, y_train, y_test)

for name in glob.glob('placas/*'):
    car_image = imread(name)
    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(1,1),0)
    ret,binaria = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    threshold_value = threshold_otsu(gray)
    binary_car_image = gray > threshold_value
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(gray, cmap="gray")
    # ax2.imshow(binary_car_image, cmap="gray")
    
    # this gets all the connected regions and groups them together
    label_image = measure.label(binary_car_image)
    
    plate_dimensions = (0.06*label_image.shape[0], 0.1*label_image.shape[0], 0.06*label_image.shape[1], 0.09*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []
    
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(car_image, cmap="gray")
    
    
    for region in regionprops(label_image):
        if region.area <50:
            continue
    
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                             max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",linewidth=2, fill=False)
            ax1.add_patch(rectBorder)            
    plt.show()
    

    for i in range(len(plate_like_objects)):   
        license_plate = np.invert(plate_like_objects[i])
        labelled_plate = measure.label(license_plate)
        character_dimensions = (0.3*license_plate.shape[0], 0.55*license_plate.shape[0], 0.08*license_plate.shape[1], 0.25*license_plate.shape[1])
        min_height, max_height, min_width, max_width = character_dimensions
    
        fig, ax1 = plt.subplots(1)
        ax1.imshow(license_plate, cmap="gray")
    
        characters = []
        counter=0
        column_list = []
        for regions in regionprops(labelled_plate):
            y0, x0, y1, x1 = regions.bbox
            y0 -= 1
            region_height = y1 - y0
            region_width = x1 - x0
    
            if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                roi = license_plate[y0:y1, x0:x1]
    
                rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                            linewidth=2, fill=False)
                ax1.add_patch(rect_border)
    
                # resize the characters to 20X20 and then append each character into the characters list
                resized_char = resize(roi, (20, 20))
                image_data = resized_char.flatten()
                # Tradon = radon(resized_char, theta=np.arange(0,210,30), circle=True).flatten()
                # Firma = resized_char.flatten()
                # Hu = cv2.HuMoments(cv2.moments(resized_char)).flatten()
                # image_data = np.concatenate((Tradon, Firma,Hu), axis=None)
                characters.append(image_data)
        plt.show()
        
        X = np.array(characters)
        X = scaler.transform(X)
        knn_pred = list(knn.predict(X))
        msv_pred = list(msv.predict(X))
        clf_pred = list(clf.predict(X))
        
        letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
                    'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N','O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']
        nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        for letra in range(len(knn_pred)):
            for x in range(len(nums)):
                if knn_pred[letra] == nums[x]:
                    knn_pred[letra] = letters[x]
                if msv_pred[letra] == nums[x]:
                    msv_pred[letra] = letters[x]
                if clf_pred[letra] == nums[x]:
                    clf_pred[letra] = letters[x]
        
        Imp_Placa(knn_pred,msv_pred,clf_pred)
        
        

