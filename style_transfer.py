import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def load_and_process_image(image_path, max_dim=512):
    img = Image.open(image_path).convert("RGB")
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return tf.image.convert_image_dtype(img, tf.float32)
content_path = 'shinchan.jpg'
style_path = 'style.jpg'
content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
def vgg_layers(layer_names):
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.vgg.trainable = False
        self.style_layers = style_layers
        self.content_layers = content_layers
    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]
        style_outputs = [gram_matrix(output) for output in style_outputs]
        content_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
extractor = StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
image = tf.Variable(content_image, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
style_weight = 1e-2
content_weight = 1e4
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        style_loss = tf.add_n([
            tf.reduce_mean((outputs['style'][name] - style_targets[name]) ** 2)
            for name in style_targets
        ])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([
            tf.reduce_mean((outputs['content'][name] - content_targets[name]) ** 2)
            for name in content_targets
        ])
        content_loss *= content_weight / num_content_layers

        loss = style_loss + content_loss

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))
epochs = 5
steps_per_epoch = 20
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        train_step(image)
    print(f"Epoch {epoch+1} completed")
final_img = image.numpy()[0]
final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)
if final_img.shape[-1] == 1:
    final_img = np.repeat(final_img, 3, axis=-1)
output_image_path = 'stylized_shinchan.jpg'
Image.fromarray(final_img).save(output_image_path)

plt.imshow(final_img)
plt.axis('off')
plt.title("Stylized Shinchan")
plt.show()
print(f"Stylized image saved as {output_image_path}")
