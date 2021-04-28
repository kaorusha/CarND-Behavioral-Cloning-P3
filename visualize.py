from keras.models import load_model

model = load_model('model.h5')
print(model.summary())

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='examples/model_plot.png', show_shapes=True, show_layer_names=True)