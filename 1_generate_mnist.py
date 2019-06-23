# Uses a minimax loss and a simple generator descriptor networks to generate
# images similar to mnist data.
#   Roughly follows: https://medium.com/datadriveninvestor/generative-adversarial-network-gan-using-keras-ce1c05cfdfd3
#   https://arxiv.org/pdf/1406.2661.pdf

# import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm #progress bar
import cv2

def show( im_set ):
    """ im_set.shape : Nx28x28 """
    assert len(im_set.shape)==3

    for i in range( im_set.shape[0] ):
        cv2.imshow( 'win', ( (im_set[i,:,:] * 127.5)+127.5 ).astype('uint8') )
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyWindow('win')

def grid_imshow( im_set, outfname=None ):
    """ im_set.shape : Nx28x28 """
    assert len(im_set.shape)==3

    N = im_set.shape[0]
    assert int(np.sqrt(N)) == np.sqrt(N)   # better be a perfect square like 9, 25, 36 etc
    N_sqrt = int( np.sqrt( N ) )


    S = []
    for r in range( N_sqrt ):
        RR = []
        for c in range( N_sqrt ):
            i = r*N_sqrt + c
            im_x = ( (im_set[i,:,:] * 127.5)+127.5 ).astype('uint8')
            RR.append( im_x )
            RR.append( np.ones(   (im_x.shape[0], 10 ), dtype='uint8' )*255   )

        # import code
        # code.interact( local=locals() )
        full_row = np.concatenate( RR, axis=1 )
        S.append( full_row )
        S.append( np.ones( (10, full_row.shape[1]), dtype='uint8' )*255 )
    out = np.concatenate( S, axis=0 )

    if outfname is None:
        cv2.imshow( 'win', out )
        cv2.waitKey(0)
    else:
        print( 'Write file: ', outfname )
        cv2.imwrite( outfname, out )

    return out



#---
#--- Data
#---
def load_data():
    # Load
    print 'Load mnist data'
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print 'x_train.shape=', x_train.shape, 'y_train.shape=', y_train.shape
    print 'x_test.shape=', x_test.shape, 'y_test.shape=', y_test.shape

    # Data range --> -0.5 to 0.5
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_test = (x_test.astype(np.float32) - 127.5)/127.5

    # Reshape; Nx32x32 ---> Nx784
    x_train = x_train.reshape( -1, x_train.shape[1]*x_train.shape[2] )
    x_test  = x_test.reshape( -1, x_test.shape[1]*x_test.shape[2]    )

    return x_train

x_train = load_data()
batch_size = 300
n_batches = int( x_train.shape[0]/batch_size )
print '---batch_size=', batch_size, '\tn_batches=', n_batches


#---
#--- Generator Network
#---
gen_model = keras.models.Sequential()
gen_model.add( keras.layers.Dense(256, activation=None, input_shape=(100,) ) )
gen_model.add( keras.layers.LeakyReLU(alpha=0.2))
gen_model.add( keras.layers.BatchNormalization() )
gen_model.add( keras.layers.Dense(512, activation=None ) )
gen_model.add( keras.layers.LeakyReLU(alpha=0.2))
gen_model.add( keras.layers.BatchNormalization() )
gen_model.add( keras.layers.Dense(1024, activation=None ) )
gen_model.add( keras.layers.LeakyReLU(alpha=0.2))
gen_model.add( keras.layers.BatchNormalization() )
gen_model.add( keras.layers.Dense(784, activation='tanh'  ) ) #28x28 ==> 784
gen_model.compile(optimizer='adam', loss='binary_crossentropy')
gen_model.summary()


#---
#--- Discriminator Network
#---
dis_model = keras.models.Sequential()
dis_model.add( keras.layers.Dense(1024, activation=None, input_shape=(784,) ) )
dis_model.add( keras.layers.LeakyReLU(alpha=0.2))
dis_model.add( keras.layers.BatchNormalization() )
# dis_model.add( keras.layers.Dropout(0.3))
dis_model.add( keras.layers.Dense(512, activation=None) )
dis_model.add( keras.layers.LeakyReLU(alpha=0.2))
dis_model.add( keras.layers.BatchNormalization() )
# dis_model.add( keras.layers.Dropout(0.3))
dis_model.add( keras.layers.Dense(256, activation=None ) )
dis_model.add( keras.layers.LeakyReLU(alpha=0.2))
dis_model.add( keras.layers.BatchNormalization() )
# dis_model.add( keras.layers.Dropout(0.3))
dis_model.add( keras.layers.Dense(1, activation='sigmoid' ) ) #28x28 ==> 784
dis_model.compile(optimizer='adam', loss='binary_crossentropy')
dis_model.summary()


#---
#--- Make GAN
#---
#       Practical tips: Use tanh activation at generator, use batch norm, relu is bad for GANs, use LeakyReLU
z = keras.layers.Input(shape=(100,))
x = gen_model( z )
verd = dis_model( x )
gan = keras.models.Model( inputs = z, outputs=verd )
gan.compile(optimizer='adam', loss='binary_crossentropy')
gan.summary()
# quit()

#---
#--- Training
#---
for e in range( 400 ): #epochs
    print( 'epoch#', e )
    for b in tqdm( range( n_batches ) ):
        # print( 'batch#', b )
        # Generate fake images (using gen_model). fakes will have zero label
        fakes = gen_model.predict( np.random.normal(0, 1, [batch_size, 100] ) ) #600 fakes
        y_fakes = np.zeros( batch_size )

        # Get hold of 600 real images. reals will have one label
        # reals = x_train[ b*batch_size:(b+1)*batch_size, :]
        reals = x_train[ np.random.randint(low=0,high=x_train.shape[0], size=batch_size) ]
        y_reals = np.ones( batch_size )*0.9

        # Concatenate above 2 datas.
        XX = np.concatenate( (fakes, reals) )
        YY = np.concatenate( (y_fakes, y_reals) )

        # Pretrain the descriminator
        dis_model.trainable = True
        dis_model.train_on_batch(XX, YY)


        # train the GAN keeping descriminator as const
        noise= np.random.normal(0,1, [batch_size, 100])
        y_gen = np.ones(batch_size)
        dis_model.trainable=False

        gan.train_on_batch(noise, y_gen)


    if e % 20 == 0:
        testing_fakes = gen_model.predict( np.random.normal(0, 1, [25, 100] ) )
        grid_imshow( testing_fakes.reshape( -1, 28, 28 ), './generated_mnist/epoch_%d.png' %(e)  )

#---
#--- Test - Lets generate a few fakes
#---
testing_fakes = gen_model.predict( np.random.normal(0, 1, [25, 100] ) )
# show( testing_fakes.reshape( -1, 28, 28 ) )
grid_imshow( testing_fakes.reshape( -1, 28, 28 ) )
