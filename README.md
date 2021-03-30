# Convolutional neural network for car model classification

Prova finale per la laurea triennale in Ingegneria Informatica presso il Dipartimento di Ingegneria ”Enzo Ferrari” - Università degli studi di Modena e Reggio Emilia
<br><br>
Relatore: Prof.ssa Rita Cucchiara
<br>
Tutor: Ing. Andrea Palazzi

Convolutional neural networks are a very powerful and eﬃcient approach for image recognition and classiﬁcation problems, but not olny, indeed they can be also integrated in more complex systems to solve more challenging and elaborated problems. An interesting and explanatory example of how a CNN works and how it can cooperate with other techniques for achieving more complex results is given by ”Semi parametrics novel view point synthesis” project developed by Unimore Aimage.
Lab.

The goal of the ”Semi parametrics novel view point synthesis” project is synthesize realistic novel views of an object from a single image. For doing this is used a semi-parametric approach in which appearance and shape are disentangled and the non-parametric visual hints act as prior information for a deep parametric model.
The framework receives as input only a single monocular image and the desired viewpoint and returns a novel viewpoint of the object represented in the image like is possible to see in the examples below. To operate in a more real-world scenario the framework doesn’t operate on synthetic data but takes the images from an existing datasets for 3D object detection.

![image](https://user-images.githubusercontent.com/45602824/112963975-2c30ca80-9148-11eb-9f2d-6c08efb7d6b6.png)

In order to analyse and discus some technical aspects of CNN this work compare some possible implementation of a convolutional network for the classiﬁcation of the object class. To simplify the analysis the net works only on the vehicle class but the same approaches and algorithms can be easily extended also at the other object classes.
The net presented was developed in Python using PyTorch library. For this work have been used 6.036 images taken from the Vehicle class of the ”PASCAL3D+” dataset highlighted in the figure below. ”PASCAL3D+” dataset is a novel and challenging dataset for 3D object detection and pose estimation that presents 12 classes of object including ”Vehicles”.
Each class present 10 sub-classes depicted in Figure 5.4 covering the intraclass variation for the object. In the case of vehicles the classes are:

![image](https://user-images.githubusercontent.com/45602824/112964024-394db980-9148-11eb-84aa-42b0b605a17a.png)

![image](https://user-images.githubusercontent.com/45602824/112964191-639f7700-9148-11eb-8bf2-1e7a095a95f9.png)

![image](https://user-images.githubusercontent.com/45602824/112964460-a5c8b880-9148-11eb-8e29-7f7b5befdc18.png)

