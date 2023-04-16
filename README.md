# Machine Learning II

## Práctica de Deep Learning

Este repositorio se ha creado para el curso de Machine Learning II del Máster de Big Data. Tecnología y Analítica Avanzada. Proyecto realizado por:

|Nombre               | Correo                             |
|---------------------|------------------------------------|
|Carlota Monedero     | carlotamoh@alu.comillas.edu        |
|José Manuel Vega     | josemanuel.vega@alu.comillas.edu   |
|Jorge Ayuso          | jorgeayusomartinez@alu.comillas.edu|
|Javier Gisbert       | 202208977@alu.comillas.edu         |
|Diego Sanz-Gadea     | d.sanz-gadea@alu.icai.comillas.edu |

En esta práctica, se exploran algoritmos de Machine Learning y Deep Learning para clasificar imágenes. Para ello, se utilizan dos técnicas: la bolsa de palabras (Bag-of-Words o BoW) y las redes neuronales convolucionales (Convolutional Neural Networks o CNN).

BoW es una técnica de Machine Learning tradicional que se basa en la extracción de características previas a la clasificación de imágenes. Visual BoW (Bolsa de Palabras visual) extrae vectores de características mediante filtros y los agrupa en clusters a través de una técnica de clustering no supervisado. Los vectores se reducen a un vocabulario de m clusters y se cuentan para crear una firma única para cada imagen. Esto es similar a la comparación de documentos dentro del modelo de BoW, donde se crea un histograma de frecuencia de palabras. Los histogramas normalizados de cada imagen se utilizan como entrada en el modelo de clasificación junto con la etiqueta de clase correspondiente.

Por otro lado, las CNN son una técnica de Deep Learning que se inspira en el funcionamiento del sistema visual humano. Las CNN aprenden automáticamente las características relevantes de las imágenes a través de capas convolucionales, y luego utilizan estas características para clasificar las imágenes. 

Además de las técnicas de BoW y CNN mencionadas, de manera adicional hemos entrenado un modelo de Vision Transformers. Los transformers son una técnica de Deep Learning que utiliza una arquitectura basada en atención para procesar secuencias de datos, como texto o imágenes. En el caso de las imágenes, los transformers aprenden a atender a diferentes partes de la imagen para extraer características relevantes. Esto se puede entender como un proceso análogo a la atención humana, en el que prestamos más atención a ciertas partes de una imagen que a otras.

Una vez que se han clasificado las imágenes utilizando estas técnicas, se procede a generar imágenes sintéticas utilizando una red generativa antagónica profunda,Deep Convolutional Generative Adversarial Network (DCGAN). Esta técnica de Deep Learning permite generar imágenes sintéticas que parecen auténticas a ojos de observadores humanos, lo que la convierte en una técnica muy interesante para la creación de contenido generado automáticamente.

En resumen, en esta práctica se han explorado técnicas de Machine Learning y Deep Learning para la clasificación de imágenes, y se ha utilizado una red generativa antagónica profunda para generar imágenes sintéticas.
