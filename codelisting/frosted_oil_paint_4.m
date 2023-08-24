%convert the image to the YCbCr colour space
YCC = rgb2ycbcr(I);
%compute superpixel segmentation
[out, L, boundary_lines] = superpixel(YCC,seg_count);
%obtain the Y channel
channel_bright = squeeze(out(:,:,1));
%obtain the mean of the channel
mean_bright = mean(channel_bright,'all');
%obtain the median of the channel
median_bright = median(channel_bright,'all');
%duplicate the Y channel
A = channel_bright;
%get all segments from Y channel > mean/median -> thresholding
%creates a black and white image where glass (background) and oil paint (foreground) sections are defined
A(channel_bright < mean_bright) = 0; %all dark areas are set to background
A(channel_bright >= mean_bright) = 1; %all light areas are set to foreground

