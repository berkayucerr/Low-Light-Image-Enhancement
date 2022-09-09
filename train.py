from utils import PSNR
def train(model,dataset,epoch):
    model.compile(loss = 'mean_squared_error',optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[PSNR])
    model.fit(dataset,epochs=epoch)