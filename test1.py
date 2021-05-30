import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model

col= ['Cardiomegaly','Emphysema','Effusion','Infiltration',
      'Mass','Nodule','Atelectasis','Pneumothorax',
      'Pleural_Thickening','Fibrosis','Edema','Consolidation','Normal']

# creating index to class dictionary
idx_class={i:c for i,c in enumerate(col)}

model = load_model('13-class-model.h5')
def classify(image):
    
    sample_image = image  # batch 0 so that returns ( h , w , c) for the image, without the batch dimention
    #sample_label = label # takes batch of xs and ys # x= train_data.next() -> x[0].shape -> 32,224,224,3
    
    sample_image_processed = np.expand_dims(sample_image, axis=0) # adding back the batch dimention
    
    activations = vis_model.predict(sample_image_processed) # the output of each layer -features-
    
    pred_label = np.argmax( model.predict(sample_image_processed) , axis=-1 )[0]
    pred_label = idx_class[pred_label]
    
    print(activations[0].shape)
    sample_activation = activations[0] [0 , : , : , :3] # taking the first output , for image of batch 0, and for the last layer #16 , --> (h,w)
    
    sample_activation-=sample_activation.mean()
    sample_activation/=sample_activation.std()
    
    sample_activation *=255
    sample_activation = np.clip( sample_activation , 0 , 255 ).astype(np.uint8)
    
    f,ax = plt.subplots(1,2, figsize=(15,8))

    ax[0].imshow(sample_image)
    ax[0].set_title(f"Predicted label: {pred_label}")
    ax[0].axis('off')
    
    ax[1].imshow(sample_activation)
    ax[1].set_title("Random feature map")
    ax[1].axis('off')
 
    plt.tight_layout()
    plt.show()
  
    return activations

outputs = [ layer.output for layer in model.layers[1:] ] # all layers except the input layer

# Define a new model that generates the above output
vis_model = Model(model.input , outputs)


from tensorflow.keras.preprocessing import image
#2- setting the path of the image
path='New Folder/normal.jpg'
#3- uploading the image into a variable

img= image.load_img( path , target_size=( 224,224 ) )
# don't forget the target size the model is expecting
#4- processing the image variable to suit the model

x= image.img_to_array( img )

c=classify(img)