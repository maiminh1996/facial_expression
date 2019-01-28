"""
├── data
│   ├── example_submission.csv
│   ├── fer2013
│   │   ├── fer2013.bib
│   │   └── README
│   ├── fer2013.csv
│   └── fer2013.tar.gz
├── E1454.full.pdf
├── fer2013.csv
├── kaggle_test_data.zip
├── main.py
└── src
    ├── analyse_video.py
    ├── haarcascade_eye.xml
    ├── haarcascade_frontalface_default.xml
    ├── main.py
    └── train_SVM.py

Learn to 
	Use pandas with csv
	Add path to projet
	How to create train and test data with sklearn

"""
import pandas as pd 
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import os, sys
file_dir = os.path.dirname(os.path.abspath("/home/minh/Documents/Artificial-Intelligence/computer_vision/face_detect_opencv_python/data"))
package_dir_a = os.path.join(file_dir, 'data')
sys.path.append(package_dir_a) #sys.path.insert(0, package_dir_a)


if __name__ == '__main__':
	# print("\n".join(sys.path))
	# f = open("../data/fer2013.csv")
	with open(package_dir_a + "/fer2013.csv") as f: # don't need to f.close()
		dataframe = pd.read_csv(f)
	
	# check the data frame info
	check = 0
	if check:
		print(dataframe.info())
		print(dataframe.head())
		print(dataframe["Usage"].unique())
		print(len(dataframe['Usage'].unique().tolist()))
		print(dataframe["emotion"].unique())
		print(len(dataframe['emotion'].unique().tolist()))
		print(len(dataframe['pixels'].unique().tolist()))
	emotion1 = np.array(dataframe["emotion"])
	emotion = np.zeros((35887, 7))
	emotion[np.arange(len(dataframe["pixels"])), emotion1] = 1
	emotion = emotion.reshape([35887, 7])


	pixels = [np.reshape(np.fromstring(dataframe["pixels"][i], dtype=int, sep=' ')/255, [48, 48, 1]) for i in range(len(dataframe["pixels"]))] 
	print(pixels[0])
	typeUsage = dataframe["Usage"]
	
	indexSearch = 0
	if indexSearch:
		a = [i for i in range(len(dataframe["pixels"])) if typeUsage[i]=='PublicTest']
		print(a[0]) # --> 28709
		b = [i for i in range(len(dataframe["pixels"])) if typeUsage[i]=='PrivateTest']
		print(b[0]) # --> 32298
		emoTrain, imgTrain, typeTrain = [emotion[:a[0]], pixels[:a[0]], typeUsage[:a[0]]]
		emoPubTest, imgPubTest, typePubTest = [emotion[a[0]:b[0]], pixels[a[0]:b[0]], typeUsage[a[0]:b[0]]]
		emoPriTest, imgPriTest, typePriTest = [emotion[b[0]:], pixels[b[0]:], typeUsage[b[0]:]]

	splitData = 1
	if splitData:
		test_size, seed = [0.3, 7] # 0.7 for data set
		# train_test_split: shuffle:(default=True)
		imgTrain, imgTest, emoTrain, emoTest = model_selection.train_test_split(pixels, emotion, test_size=test_size, random_state=seed)
		print(len(pixels))
		print(len(imgTrain))
		print(np.shape(pixels[:2]))


	# print("hehe", len(dataframe["pixels"]))
	# print("huhu", len(dataframe["emotion"]))
	# img1 = np.reshape(pixels[0], [48, 48])
	# plt.imshow(img1)
	# plt.show()
	
