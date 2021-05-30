def face_rec():
	import face_recognition
	import os
	from numpy import load
	import time
	import picamera

	with picamera.PiCamera() as camera:
	   camera.resolution = (1024, 768)
 	   camera.start_preview()
 	   # Camera warm-up time
 	   time.sleep(4)
  	   camera.capture('foo.jpg')
	# Load the images into numpy arrays
	authorized_encodings=load('embeddings.npy', allow_pickle=True)
	unknown_images =face_recognition.load_image_file('foo.jpg')
	# Get the face encodings for each face in each image file
	# Since there could be more than one face in each image, it returns a list of encodings.

	try:
	    unknown_encodings =face_recognition.face_encodings(unknown_images)

	except IndexError:
 	   print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
 	   quit()
    
	# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
	results = [ face_recognition.compare_faces(authorized_encodings, unk) for unk in unknown_encodings]
	new_res= [ result[i]  for result in results for i in result ]
	out= True in new_res
	print("Is the person authorized? {}".format( True in new_res ))

	return out
	
