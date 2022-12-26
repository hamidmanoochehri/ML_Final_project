# ==============================================================================
# ===========================imports & global params============================
# ==============================================================================
# %reset -f
import cv2
import numpy             as np
import matplotlib.pyplot as plt
import argparse
import os
from imutils                             import paths
from sklearn.preprocessing               import LabelBinarizer
from sklearn.model_selection             import train_test_split     as Split
from sklearn.metrics                     import classification_report as creport
from sklearn.metrics                     import confusion_matrix     as cmatrix
from tensorflow.keras                     import layers
from tensorflow.keras                     import optimizers           as opts
from tensorflow.keras.models              import Model
from tensorflow.keras.applications        import VGG16
from tensorflow.keras.utils               import to_categorical       as cats
from tensorflow.keras.preprocessing.image import ImageDataGenerator

start_exp = 13
figA      = list( '0'*start_exp )
axA      = list( '0'*start_exp )
figB      = list( '0'*start_exp )
axB      = list( '0'*start_exp )
preds    = list( '0'*start_exp )
actuals  = list( '0'*start_exp )
cr       = list( '0'*start_exp )
cm       = list( '0'*start_exp )
total    = list( '0'*start_exp )
accuracy = list( '0'*start_exp )
figB      = list( '0'*start_exp )
axB      = list( '0'*start_exp )
# experiment loop
for exp in range(start_exp,14):
  print('********EXPERIMENT '+str(exp)+'*********')
  # Params
  colab_use  = True
  sampleNo   = 10
  epochs     = 10
  batch_size = 8
  rotation   = 15
  init_lr    = .001
  test_size  = .1

  # ==============================================================================
  # ===============================loading gdrive=================================
  # ==============================================================================
  if colab_use:
    # mounting drive (for colab)
    from google.colab import drive
    drive.mount('/Mounted')

  path     = 'dataset' +str(exp)+ '/brain_tumor_dataset/'
  savepath = 'results/dataset' +str(exp)+ '/'
  Paths = list(paths.list_images(path))
  print('data dirs:'         , os.listdir(path))
  print('no of all images: ' , len(Paths))

  # ==============================================================================
  # ===============================images & labels================================
  # ==============================================================================
  images = [] # initializing images list
  labels = [] # initializing labels list
  for p in Paths:
    l = p.split(os.path.sep)[-2] # label portion
    i = cv2.resize(cv2.imread(p), (224, 224))
    images.append(i) # creating datalist of images
    labels.append(l) # creating datalist of labels

  # showing some samples of images
  plt.rcParams['figure.figsize'] = (40,16)
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap'] = 'gray'
  figA.append( plt.figure() )
  axA.append( [] )
  print(axA)
  for i in range(sampleNo):
    axA[exp].append(figA[exp].add_subplot(2,int(sampleNo/2),i+1))
    axA[exp][i].imshow(images[i])
    axA[exp][i].set_title('Data Sample ' + str(i) , fontsize=30)
  plt.savefig(savepath+'sample_images.png')
  # prepariong the dataset dataset into train & test

  images   = np.array(images)/255.0
  labels   = np.array(labels)
  binLabel = LabelBinarizer()
  labels   = cats(binLabel.fit_transform(labels))
  (x_train , x_test , y_train , y_test) = Split(images , labels , test_size=test_size , random_state=42 , stratify=labels)

  #train data generator
  datagen = ImageDataGenerator(fill_mode='nearest' , rotation_range=rotation)

  # ==============================================================================
  # ===================build & compile & summary of the model=====================
  # ==============================================================================
  model0  = VGG16(weights='imagenet' , input_tensor=layers.Input(shape=(224,224,3)),
                  include_top=False)
  input0  = model0.input
  output0 = model0.output
  output0 = layers.AveragePooling2D(pool_size=(4 , 4))(output0)
  output0 = layers.Flatten(name="flatten")(output0)
  output0 = layers.Dense(64 , activation="relu")(output0)
  output0 = layers.Dropout(.5)(output0)
  output0 = layers.Dense(2 , activation="softmax")(output0)

  # compiling our model
  for i in model0.layers:
      i.trainable = False
  model = Model(inputs=input0 , outputs=output0)
  opt   = opts.Adam(learning_rate=init_lr)
  model.compile(metrics=['accuracy'] , loss='binary_crossentropy' , optimizer=opt)

  # output the model architecture summary
  model.summary()

  # ==============================================================================
  # ===============================train the model================================
  # ==============================================================================
  steps     = len(x_train)//batch_size
  val_data  = (x_train , y_test)
  val_steps = len(x_test) //batch_size
  history   = model.fit( datagen.flow( x_train , y_train,
                                    batch_size       = batch_size),
                                    steps_per_epoch  = steps,
                                    validation_data  = (x_test , y_test),
                                    validation_steps = val_steps,
                                    epochs           = epochs             )

  # ==============================================================================
  # =============================evaluate the model===============================
  # ==============================================================================
  figname_loss  = 'loss_dataset'+str(exp)+'.png'
  figname_accu  = 'accu_dataset'+str(exp)+'.png'

  preds.append( np.argmax(model.predict(x_test , batch_size=batch_size) , axis=1) )
  actuals.append( np.argmax(y_test , axis=1) )

  # Print Classification report and Confusion matrix
  cr.append( creport(actuals[exp] , preds[exp], target_names=binLabel.classes_) ) # generating classification report
  cm.append( cmatrix(actuals[exp] , preds[exp]) ) # generating confusion matrix
  print('---------Classification Matrix---------\n', cr[exp])
  print('---------Confusion     Matrix---------\n', cm[exp])

  # Final accuracy of our model
  total.append( sum(sum(cm[exp])) )
  accuracy.append( ( cm[exp][0,0] + cm[exp][1,1] )/total[exp] )
  print("Accuracy: {:.4f}".format(accuracy[exp]))

  # Plotting the results
  plt.rcParams['figure.figsize']        = (20,8)
  plt.rcParams['image.interpolation'] = 'nearest'
  plt.rcParams['image.cmap']          = 'gray'
  plt.style.use("ggplot")

  figB.append( plt.figure() )
  axB.append( [] )

  axB[exp].append(figB[exp].add_subplot(1,2,1))
  axB[exp][0].plot(np.arange(0,epochs), history.history["loss"]     , label= "training loss")
  axB[exp][0].plot(np.arange(0,epochs), history.history["val_loss"] , label= "validation loss"  )
  axB[exp][0].set_title('training and validation loss'              , fontsize=15        )
  plt.xlabel("epoch", fontsize=14)
  plt.ylabel("loss" , fontsize=14)
  plt.legend(loc="upper right")
  axB[exp][0].set_ylim([0,1])
  plt.savefig(savepath + figname_loss)

  axB[exp].append(figB[exp].add_subplot(1,2,2))
  axB[exp][1].plot(np.arange(0,epochs) , history.history["accuracy"]     , label= "training accuracy")
  axB[exp][1].plot(np.arange(0,epochs) , history.history["val_accuracy"] , label= "validation accuracy"  )
  axB[exp][1].set_title('training and validation accuracy'               , fontsize=15       )
  plt.xlabel("epoch"   , fontsize=14)
  plt.ylabel("accuracy", fontsize=14)
  plt.legend(loc="upper left")
  axB[exp][1].set_ylim([0,1])
  figB[exp].savefig(savepath + figname_accu)
