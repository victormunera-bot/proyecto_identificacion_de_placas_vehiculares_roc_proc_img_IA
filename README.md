
# Identificación de placas vehiculares en Colombia a partir de reconocimiento óptico de caracteres ROC

## Integrantes 

- Juan Manuel Aya Perlaza
- Laura María Joaqui Muñoz
- Víctor Hugo Múnera Rojas

## Descripción 
Se tiene un sistema que permite reconocer y predecir los caracteres de una placa vehicular colombiana a partir del procesamiento de una imagen inicial que contenga un vehículo y mediante el entrenamiento de algunos algoritmos de aprendizaje de máquina.

Para conseguir esto, se parte de la imagen inicial en la cual se identifica el lugar en el que se encuentra la placa a través del reconocimiento del contorno con un área y dimensiones relativas al tamaño de la imagen, luego, conociendo el lugar donde está la placa se toma este nuevo recorte de imagen y, mediante el mismo algoritmo, se detectan cada una de las letras/números presentes en ella. Esto permite generar 6 nuevos recortes de imagen, cada uno pasará por un modelo de Aprendizaje de Máquina previamente entrenado para poder predecir a qué carácter alfanumérico hace referencia. 

## Nota

**Para el correcto funcionamiento del código principal hace falta la carga de dos carpetas con imágenes confidenciales ya que contienen datos privados acerca de los propietarios de los vehiculos en las imagenes a procesar.**  

## Imágenes relacionadas 

A continuación, se adjuntan imágenes relacionadas con el proceso a realizar en este proyecto:

![image](https://upload.wikimedia.org/wikipedia/commons/9/9c/California_license_plate_ANPR.png?1607060389900 )


![enter image description here](https://kipod.com/wp-content/uploads/2016/12/imageedit_3_2524798281-820x400.jpg)



