%https://www.imageeprocessing.com/2011/06/glassy-effect-in-matlab.html
%computing glassy version of the image
%input:
%I: RGB Oil Painted image
%threshold: black and white image where glass and oil paint sections are defined
%filter_half_width: half the width of the neighbourhood being filtered
%Output: glassy image strongly weighted the further away fronm the painted section it gets
function output = glass(I,filter_half_width, threshold)
    %erode threshold for glass effect to be stronger when closer to the paint effect
    %disk = strel('disk',10,0);
    %threshold = imerode(threshold,disk);
    %obtain the distance matrix for weighting
    d = bwdist(threshold);
    %obtain the maximum possible distance in the matrix
    max_d = max(d,[],'all');
    %normalise the distance matrix values
    d = d./max_d;
    %create a copy of the image
    I_copy = I;
    %obtain dimensions of image    
    [rx, gy, bz] = size(I_copy);
    output=uint8(zeros([rx,gy,bz]));
    %pad the dimension 
    rx = rx + 2*filter_half_width;
    gy = gy + 2*filter_half_width;
    %obtain each layer
    layerR = squeeze(I_copy(:,:,1));
    layerG = squeeze(I_copy(:,:,2));
    layerB = squeeze(I_copy(:,:,3));
    %mirror pad each layer
    newR = padarray(layerR,[filter_half_width filter_half_width],'symmetric','both');
    newG = padarray(layerG,[filter_half_width filter_half_width],'symmetric','both');
    newB = padarray(layerB,[filter_half_width filter_half_width],'symmetric','both');
    %recombine layers for padded version of input image
    I_copy = cat(3, newR, newG, newB);
    %loop through the unpadded input image
    for i=1+filter_half_width:rx-filter_half_width
        for j=1+filter_half_width:gy-filter_half_width
            %normalise the weights
            ratio_d = d(i-filter_half_width,j-filter_half_width);
            %use the weight to set the size of the filter
            filter_size = floor(filter_half_width*ratio_d);
            %obtain the neighbourhood defined by the weighted filter_size
            mask = I_copy(i-filter_size:i+filter_size,j-filter_size:j+filter_size,:);
            %Select a pixel value from the mask neighborhood
            %rand(1) gets a 1x1 matrix with values from 0-1
            %the closer to the oil painted region, the weaker the filter gets
            x_2 = ceil(rand(1)*filter_size)+1;
            y_2 = ceil(rand(1)*filter_size)+1;
            %apply the filter value
            output(i-filter_half_width,j-filter_half_width,:) = mask(x_2,y_2,:);
        end
    end
end
