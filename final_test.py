def classifier(path='x-ray.jpg')
  import tensorflow as tf
  import numpy as np
  from tensorflow.keras.preprocessing import image

  img= image.load_img( path , target_size=( 224,224 ) )

  x= image.img_to_array( img )
  x= np.expand_dims(x , axis=0)

  col= ['Cardiomegaly','Emphysema','Effusion','Infiltration',
      'Mass','Nodule','Atelectasis','Pneumothorax',
      'Pleural_Thickening','Fibrosis','Edema','Consolidation','Normal']
  # creating index to class dictionary
  idx_class={i:c for i,c in enumerate(col)}



  interpreter= tf.lite.Interpreter(model_path='litemodel.tflite')
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  interpreter.set_tensor(input_details[0]['index'], x )

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.

  output_data = interpreter.get_tensor(output_details[0]['index'])
  class= idx_class[output_data[0].argmax()]
  print( f'class= {class}'  )
  
  return( class ) 