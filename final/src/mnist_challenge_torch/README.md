#

## tf.nn.max_pool函数

h : 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch_size, height, width, channels]这样的shape

k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加

## 
kernel_size – the size of the window to take a max over

stride – the stride of the window. Default value is kernel_size

padding – implicit zero padding to be added on both sides

dilation – a parameter that controls the stride of elements in the window

return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later

ceil_mode – when True, will use ceil instead of floor to compute the output shape