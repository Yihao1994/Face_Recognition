# Face_Recognition
The Face Recognition's application basing on the FaceNet, which is trained in a deep inception network. The well-trained model data are stored in the .csv files in the 'weights' folder, and can be loaded by the function ___load_weights_from_FaceNet___.  

There is one thing need to be mentioned about the operating environment. These scripts here were executing on the tensorflow-gpu-1.8.0. After installing the tensorflow-gpu-1.8.0, you have to activate tensorflow environment by pressing the 'activate tensorflow' into the Anaconda prompt at first, and then open the Spyder from the Anaconda prompt. And furthermore, the version of **keras** using here is **keras-2.2.2**, and a lower version of keras will probably generate some unknown errors. Then so far, the environment problem shall be fixed. However, when you try to load the model in line 60, it might will take 1-2 mins, as the size of the model is not small.  

There are two floders in the folder 'images':  

The first folder is called 'Database', is where to put the anchor image inside. Due to the lmitation of the model, the resolution of your anchor image in the database shall be strictly **96*96**;

While on the other hand, for the second folder 'FR_test', with the help of some programming technique, the limitation of your test image will be not that strict. As long as they are squared images, then that shall be all right. The reason for using squared testing image is to match the **96*96** anchor image, just in case of too mcuh distortion.  

From line 62-87 is where to load the database. The database are all counting on what you put into the 'database' folder that I mentioned before. As you might can see, that I put several "Yihao_?" and "Binhe_?" inside. The reason for this is because of the immaturity of the model, since it still cannot ignore the influence of the ambient light or the camera angle. And as a result of this, several "Yihao_?" or "Binhe_?" shall be judged as the same person, respectively. The unification of names has been described between 123 to 134.  

The mechanism of this algorithm is to find one anchor image, who has the minimum distance with the testing image, from your database. The initial minimum distance will be set to 100, and it should be updated smaller and smaller. The most important hyperparameter for this model, is the '**threshold**' putting in line 140. This value here is the maximum distance for judging if your testing image, and the anchor image who has the minimum distance with your testing image, are the same person. IF they are the same, then the right name and corresponding distance will be printed into the console window. Otherwise, it will say, there is no match image in the database.

You can try to put your image inside, and loaded it into the database by following the syntax between line 63-86 at first, and test your own image by changing the file name in line 139.  

Enjoy!  
