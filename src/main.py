from threading import Thread
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import modelCNN
import tensorflow as tf 
from pathlib import Path
from PIL import Image

class WebcamVideoCapture(Thread):
    def __init__(self, src=0):
        Thread.__init__(self)
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def run(self):
        self.update()
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class AnalyseFace():
	"""docstring for analyseFace"""
	def __init__(self, faceCascade, eyeCascade):
		self.faceCascade = faceCascade 
		self.eyeCascade = eyeCascade
		self.image, self.gray = [None, None]
		#self.Y_pred, self.sess = [None, None]
		self.Y_pred, self.sess = self.sentiment() # THIS IS IMPORTANT
	"""
	def getSuppliedValues(self):
		# Get user supplied values
		imagePath = sys.argv[1]
		cascPath = sys.argv[2]
		cascEyePath = sys.argv[3]
		return imagePath, cascPath, cascEyePath
	"""
	def update(self, img):
		self.image = img
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
	def detectFace(self):
		# Detect faces in the image
		faces = self.faceCascade.detectMultiScale(
		    self.gray,
		    scaleFactor=1.3,
		    minNeighbors=5,
		    minSize=(30, 30),
		    #flags = cv2.CV_HAAR_SCALE_IMAGE
		    )
		print ("Found {0} faces! ".format(len(faces)), faces)
		return faces
	def showFace(self, sentiment=None):
		faces = self.detectFace()
		font = cv2.FONT_HERSHEY_SIMPLEX
		for (x,y,w,h) in faces:
		    cv2.rectangle(self.image, (x,y), (x+w,y+h), (255,255,0), 2)
		    cv2.putText(self.image, sentiment, (x+int(w/2)-10, y+h), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
		cv2.imshow('img',self.image)
	def detectEye(self):
		faces = self.detectFace()
		for (x,y,w,h) in faces:
		    cv2.rectangle(self.image,(x,y),(x+w,y+h),(255,0,0),2)
		    roi_gray = self.gray[y:y+h, x:x+w]
		    roi_color = self.image[y:y+h, x:x+w]
		    eyes = self.eyeCascade.detectMultiScale(roi_gray)
		    for (ex,ey,ew,eh) in eyes:
		        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		cv2.imshow('img',self.image)
	def sentiment(self):
		self.X = tf.placeholder(tf.float32, shape=[None, 48, 48, 1], name='Input')  # for image_data
		self.is_training = tf.placeholder(tf.bool)

		YPred = modelCNN.Model(self.X, self.is_training).faceNet()
		saver = tf.train.Saver(var_list=None)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config = config)
		sess.run(tf.global_variables_initializer())
		checkpoint = "logs/model.ckpt-7"

		try:
			aaa = checkpoint + '.meta'
			my_abs_path = Path(aaa).resolve()
		except FileNotFoundError:
			print("Not yet training!")
		else:
			saver.restore(sess, checkpoint)
			#print("checkpoint: ", checkpoint)
			#print("already training!")
		# one hot to integers
		return YPred, sess
	def showSenti(self):
		
		faces = self.detectFace()
		if len(faces)!=0:

			face = faces[0]
			
			w = face[2]
			h = face[3]
			#plt.imshow(self.gray)
			#plt.show()
			image_cut = self.gray[face[1]:face[1]+h,face[0]:face[0]+w]
			image_cut = Image.fromarray(image_cut)
			#image = Image.open(filename)
			image_cut.thumbnail((48, 48), Image.ANTIALIAS)
			#image.save(filename, quality=100)
			#image_cut.resize(48,48)
			image_cut= np.array(image_cut)
			#print(np.shape(image_cut))
			#plt.imshow(image_cut)
			#plt.show()
			image_cut = np.reshape(image_cut, [1,48,48,1])
			
			Y_pred = self.sess.run(self.Y_pred, feed_dict={self.X: image_cut/255, self.is_training:False})

			Y_pred_integer = np.argmax(Y_pred) # Y_pred_integer = np.where(Y_pred==1)[0][0]
			set = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', '	']
			senti = set[Y_pred_integer]
			print(senti)
			font = cv2.FONT_HERSHEY_SIMPLEX
			
			for (x,y,w,h) in faces:
			    cv2.rectangle(self.image, (x,y), (x+w,y+h), (255,255,0), 2)
			    cv2.putText(self.image, senti, (x+int(w/2)-10, y+h), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
			cv2.imshow('img',self.image)
			# self.sess.close()
		else:
			cv2.imshow('img',self.image)


if __name__ == '__main__':
	faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
	face = AnalyseFace(faceCascade, eyeCascade) # THIS IS IMPORTANT outside of while
	# detecSenti(AnalyseFace())
	cap = WebcamVideoCapture()
	cap.start()
	while 1:
		start_time = time.time()
		# Capture frame-by-frame
		ret, frame = cap.read()
		if not ret:
			continue
		
		face.update(frame)
		face.showSenti()
		#cv2.imshow('Warped',img)
		if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
			break
		print(time.time()-start_time)
		
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

