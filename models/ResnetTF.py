import tensorflow as tf
from tensorflow.keras import layers, Model

class BasicBlock(tf.keras.layers.Layer):
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.expansion*planes, kernel_size=1, strides=stride, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out

class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=1, optimizer=None, loss=None, metrics=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()


        # Upsampling layers
        self.upconv1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_upconv1 = tf.keras.layers.BatchNormalization()
        self.upconv2 = tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_upconv2 = tf.keras.layers.BatchNormalization()
        self.upconv3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.bn_upconv3 = tf.keras.layers.BatchNormalization()
        self.conv_out = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1),activation='sigmoid')

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x):

        print('input', x.shape)

        out = tf.nn.relu(self.bn1(self.conv1(x),))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        print('blocks', out.shape)

        # out = self.avgpool(out)
        out = self.upconv1(out)
        out = tf.nn.relu(self.bn_upconv1(out))
        out = self.upconv2(out)
        out = tf.nn.relu(self.bn_upconv2(out))
        out = self.upconv3(out)
        out = tf.nn.relu(self.bn_upconv3(out))
        out = self.conv_out(out)

        print('out', out.shape)

        model = tf.keras.models.Model(inputs=[x], outputs=[out])

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics[0]
        )

        return model

def ResNet34(num_classes, optimizer, loss, metrics):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], 
        num_classes=num_classes, 
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
