#

## tf.nn.max_pool函数

h : 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch_size, height, width, channels]这样的shape

k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加

## torch.nn.MaxPool2d
kernel_size – the size of the window to take a max over

stride – the stride of the window. Default value is kernel_size

padding – implicit zero padding to be added on both sides

dilation – a parameter that controls the stride of elements in the window

return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later

ceil_mode – when True, will use ceil instead of floor to compute the output shape


## torch.nn.Conv2d()
输入：x[ batch_size, channels, height_1, width_1 ]
batch_size，一个batch中样本的个数 3
channels，通道数，也就是当前层的深度 1
height_1， 图片的高 5
width_1， 图片的宽 4

卷积操作：Conv2d[ channels, output, height_2, width_2 ]
channels，通道数，和上面保持一致，也就是当前层的深度 1
output ，输出的深度 4【需要4个filter】
height_2，卷积核的高 2
width_2，卷积核的宽 3

输出：res[ batch_size,output, height_3, width_3 ]
batch_size,，一个batch中样例的个数，同上 3
output， 输出的深度 4
height_3， 卷积结果的高度 4
width_3，卷积结果的宽度 2