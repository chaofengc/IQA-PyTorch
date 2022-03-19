# Matlab Reproduce Note

We record problems encountered during our reproduction of matlab based metrics here for your reference.

## The function [conv2](https://ww2.mathworks.cn/help/matlab/ref/conv2.html?lang=en) in Matlab

- The convolution kernel first rotate by 180 degrees and then compute convolutional results. This problem can be solved with 'rotate' or 'flip' in Pytorch.

## The function [imfilter](https://ww2.mathworks.cn/help/images/ref/imfilter.html?lang=en) in Matlab

- The Padding Option 'symmetric' use mirror reflection of its boundary to pad containing the outermost boundary of this image.

- The default Padding Option is zero-padding.

- The default Correlation and Convolution Option is 'corr', which calculate convolutions without rotate.
