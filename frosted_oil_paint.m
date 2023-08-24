%Team: Dhannya Mathew - 1706542 & Saatwik Kambadkone - 1601869
%file name
source_file = "1.jpg";
number = split(source_file,".");
input_str = "images/";
input_path = input_str + source_file;
output_str_1 = "output/frosted_oil/" + number(1) + "_oil";
output_str_2 = "output/frosted_oil/" + number(1) + "_acrylic";
output_str_3 = "output/frosted_oil/" + number(1) + "_frosted";
output_path_1 = output_str_1 + "." + number(2);
output_path_2 = output_str_2 + "." + number(2);
output_path_3 = output_str_3 + "." + number(2);
%Read in the image
I = imread(input_path);
%display the original image
figure('NumberTitle', 'off', 'Name', 'Frosted Oil Paint Process'),subplot(2,2,1), imshow(I),title('Original');
%obtain the dimensions of the image
[x,y,z] = size(I);
%setup segmentation value variable
seg_count = 30;
%setup variables for paint filter half width
paint_half_width = 5;
%setup variables for paint level
paint_level = 15;
%setup variable for glass filter half width
glass_half_width= round(x/60);

%compute superpixel segmentation
[out, ~, ~] = superpixel(I,seg_count);
%display superpixel segmentation
subplot(2,2,1), imshow(out),title('Superpixel Segmented RGB');

%convert the image to the YCbCr colour space
YCC = rgb2ycbcr(I);
%compute superpixel segmentation
[out, L, boundary_lines] = superpixel(YCC,seg_count);
%display superpixel segmentation
subplot(2,2,2), imshow(out),title('Superpixel Segmentted YCbCr');

%obtain the Y channel
channel_bright = squeeze(out(:,:,1));
%obtain the mean of the channel
mean_bright = mean(channel_bright,'all');
%obtain the median of the channel
median_bright = median(channel_bright,'all');
%display the channel
subplot(2,2,3), imshow(channel_bright),title('Superpixel Segmentated Y Channel');

%duplicate the Y channel
A = channel_bright;
%get all segments from Y channel > mean/median -> thresholding
%creates a black and white image where glass (background) and oil paint (foreground) sections are defined
A(channel_bright < mean_bright) = 0; %all dark areas are set to background
A(channel_bright >= mean_bright) = 1; %all light areas are set to foreground
%display the threshold image
subplot(2,2,4), imshow(A,[]),title('Threshold Segments'); 
%note, A is uint8 but we needed to have 1 to multiply out

%compute the oil painted image
out = paint_padded(I,paint_level,paint_half_width);
%display the oil painted image
figure('NumberTitle', 'off', 'Name', 'Oil Paint'), imshow(out);
%write out result
imwrite(out, output_path_1);
%sharpen the oil painted image
out_sharp = imsharpen(out);
%write out result
imwrite(out_sharp, output_path_2);
%display the sharpened image
figure('NumberTitle', 'off', 'Name', 'Acrylic Paint'), imshow(out_sharp);

%use threshold on painted image to extract regions
A_painted = out.* cat(3, A, A, A);
%obtain the glassey version of the image
I = glass(out,glass_half_width,logical(A));
%invert the threshold image
A = double(A);
A = uint8(imcomplement(A));
%use inverted threshold on glass image to extract regions 
A_orig = I.* cat(3, A, A, A);
%add the two images together
thresh_out = A_orig + A_painted;
%output the resulting combination
figure('NumberTitle', 'off', 'Name', 'Frosted Oil Paint'), imshow(thresh_out);
%write out result
imwrite(thresh_out, output_path_3);

%smoothing effect on boundaries=found to be an unappealing effect
%{
%Boundary extraction using morphological gradient
A = logical(A);
disk = strel('disk',1,0);
new_boundaries = imdilate(A,disk) - imerode(A,disk);
new_boundaries = bwmorph(new_boundaries,'skel',inf);
disk = strel('disk',1,0);
new_boundaries  = imdilate(new_boundaries, disk); 

figure('NumberTitle', 'off', 'Name', 'Boundary Lines'), imshow(new_boundaries);

h = gaussian_kernel(1,5,1);
layerR = squeeze(thresh_out(:,:,1));
layerG = squeeze(thresh_out(:,:,2));
layerB = squeeze(thresh_out(:,:,3));
layerR = selective_conv(layerR,new_boundaries,h);
layerG = selective_conv(layerG,new_boundaries,h);
layerB = selective_conv(layerB,new_boundaries,h);
final_border = cat(3, layerR, layerG, layerB);
figure('NumberTitle', 'off', 'Name', 'Blur'), imshow(final_border);

layerR_I = squeeze(thresh_out(:,:,1));
layerG_I = squeeze(thresh_out(:,:,2));
layerB_I = squeeze(thresh_out(:,:,3));
layerR_I(new_boundaries==1) = 0;
layerG_I(new_boundaries==1) = 0;
layerB_I(new_boundaries==1) = 0;
layerR_I = layerR_I + layerR;
layerG_I = layerG_I + layerG;
layerB_I = layerB_I + layerB;
final_I = cat(3, layerR_I, layerG_I, layerB_I);
figure('NumberTitle', 'off', 'Name', 'All'), imshow(final_I);
%}

%obtain a Gaussian smoothing filter
%input:
%K is a multiplier
%std is the standard deviation
%size is the size of the filter
%output: 
%h_out: the Gaussian smoothing filter
function h_out = gaussian_kernel(K,size,std)
    %Desiging distance matrix variables
    u = single(0:(size - 1));
    v = single(0:(size - 1));

    %compute indices to use meshgrid
    idx = find(u > size/2);
    u(idx) = u(idx) - size;
    idy = find(v > size/2);
    v(idy) = v(idy)-size;
    [V,U] = meshgrid(v,u);

    %compute the distance from every point to the origin
    h_out = K*exp(-(V.^(2) + U.^(2))/(2*std^(2))); 
    %shift to center the filter
    h_out = fftshift(h_out);
    %average each element in the filter (divide by the sum of all the elements)
    h_out = (1/sum(h_out,'all')).*h_out;
end

%https://www.imageeprocessing.com/2011/06/glassy-effect-in-matlab.html
%computing glassy version of the image
%input:
%I: RGB Oil Painted image
%threshold: black and white image where glass and oil paint sections are defined
%filter_half_width: half the width of the neighbourhood being filtered
%Output: glassy image strongly weighted the further away fronm the painted section it gets
function output = glass(I,filter_half_width, threshold)
    %erode threshold for glass effect to be stronger closer to the paint
    %effect
    %disk = strel('disk',10,0);
    %threshold = imerode(threshold,disk);
    %obtain the distance matrix for weighting
    d = bwdist(threshold);
    %ensure that areas where paint effect is has no computation necessary
    %d = d_orig; 
    %d(d_orig==0) = 1; 
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
    
    %{
    for i=1:rx-2*filter_half_width
        for j=1:gy-2*filter_half_width
            filter_size = 2*filter_half_width;
            filter_size = ceil(2*filter_half_width);
            mask = I_copy(i:i+filter_size-1,j:j+filter_size-1,:);
            
            %Select a pixel value from the neighborhood
            %rand(1) gets a 1x1 matrix with values from 0-1
            %the closer to the oil regions, the weaker the filter gets
            x_2 = ceil(rand(1)*filter_size);
            y_2 = ceil(rand(1)*filter_size);
            output(i,j,:) = mask(x_2,y_2,:);
        end
    end
    %}
end

%https://softwarebydefault.com/2013/06/29/oil-painting-cartoon-filter/
%computing oil painting version of image (with no averaging components)
%input:
%I: RGB image
%Level: Number of Histogram Bins to compute
%filter_half_width: half the width of the neighbourhood being filtered
%Output: oil painted image
function oil = paint(I,Level,filter_half_width)
    %obtain dimensions of image
    [rx, gy, bz] = size(I);
    %output formatting
    oil = uint8(zeros(rx,gy));
    %Image with intensity values respective to the RGB layers
    gray = rgb2gray(I);
    %Intensity scale the values to the desired level
    %puts it in the range of 0 to Level
    %gray now holds the intensity bin indices
    gray = round(gray*(double(Level)/255));
    
    %iterate through each pixel in the image
    for x = 1:rx
        for y = 1:gy
            %histograms setup
            intensity_counter = uint8(zeros(1,Level+1));
            imR = uint8(zeros(1,Level+1));
            imG = uint8(zeros(1,Level+1));
            imB = uint8(zeros(1,Level+1));
            %iterate through filter defined neighbourhood
            for i = (x-filter_half_width:x+filter_half_width)
                for j = (y-filter_half_width:y+filter_half_width)
                    %ensure that the index is within the range of the input
                    %image
                    if check(rx, gy, i ,j)  
                        %the histogram is indexed with the indesity values
                        %matlab starts arrays from one; therefore +1
                        intensity_idx = gray(i,j) + 1;
                        %use the index to store how many occurances of that
                        %intensity values has occured
                        intensity_counter(intensity_idx) = intensity_counter(intensity_idx) + 1;
                        %store the original image RGB values for that particular intensity
                        %value
                        imR(intensity_idx) = I(i,j,1);
                        imG(intensity_idx) = I(i,j,2);
                        imB(intensity_idx) = I(i,j,3);
                    end
                end
            end
            %find the maximum occuring intensity in intensity scale image
            [~, index] = max(intensity_counter);
            %set the output image pixel values at each layer to be the
            %maximum found RGB value in the region
            oil(x,y,1) = imR(index);
            oil(x,y,2) = imG(index);
            oil(x,y,3) = imB(index);            
        end
    end
end

%computing oil painting version of image (with padded sides and averaging)
%input:
%I: RGB image
%Level: Number of Histogram Bins to compute
%filter_half_width: half the width of the neighbourhood being filtered
%Output: oil painted image
function oil = paint_padded(I,Level,filter_half_width)   
    I_copy = I;
    %obtain dimensions of image    
    [rx, gy, bz] = size(I_copy);
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
    %output formatting
    oil = uint8(zeros(rx - 2*filter_half_width,gy - 2*filter_half_width));
    %Image with intensity values respective to the RGB layers
    grey = uint8(zeros(rx,gy));
    
    %intensity scaling the intensities in the original image to match
    %chosen level
    %puts it in the range of 0 to Level
    %gray now holds the intensity bin indices
    %loop through all the pixels in the input image
    for x = 1:rx
        for y = 1:gy
            %obtain the RGB colours at a specific pixel
            r = double(I_copy(x,y,1));
            g = double(I_copy(x,y,2));
            b = double(I_copy(x,y,3));
            sum_colour = (r + g + b)/3.0;
            %calculate the intensity value for the pixel
            grey(x,y) = (round(sum_colour * (double(Level)/255.0)));
        end
    end
    
    %iterate through each pixel in the image
    for x = 1+filter_half_width:rx-filter_half_width
        for y = 1+filter_half_width:gy-filter_half_width
            %histograms setup
            intensity_counter = zeros(1,Level+1);
            imR = zeros(1,Level+1);
            imG = zeros(1,Level+1);
            imB = zeros(1,Level+1);
            %iterate through filter defined neighbourhood
            for i = (x-filter_half_width:x+filter_half_width)
                for j = (y-filter_half_width:y+filter_half_width)                       
                    %the histogram is indexed with the indesity values
                    %matlab starts arrays from one; therefore +1
                    intensity_idx = grey(i,j) + 1;
                    %use the index to store how many occurances of that
                    %intensity values has occured
                    intensity_counter(intensity_idx) = intensity_counter(intensity_idx) + 1;
                    %store the accumulated original image RGB values for that particular intensity value
                    imR(intensity_idx) = double(imR(intensity_idx)) + double(I_copy(i,j,1));
                    imG(intensity_idx) = double(imG(intensity_idx)) + double(I_copy(i,j,2));
                    imB(intensity_idx) = double(imB(intensity_idx)) + double(I_copy(i,j,3));
                end
            end
            
            %find the maximum occuring intensity in the intensity scaling image
            [max_occurance, index] = max(intensity_counter);
            %compute the indices for the output image with respct to the
            %loop variables
            out_x = x-filter_half_width;
            out_y = y-filter_half_width;
            %set the output image pixel values at each layer to be the maximum found RGB value 
            %in the region averaged over the number of times it appears in the neighbourhood
            oil(out_x,out_y,1) = imR(index)/max_occurance;
            oil(out_x,out_y,2) = imG(index)/max_occurance;
            oil(out_x,out_y,3) = imB(index)/max_occurance;            
        end
    end
end

%superpixel segmentation
%input:
%I: RGB image
%Level: Number of Segments
%Output: segmented image, label matrix and the binary image of the boundary lines
function [segmented,L,boundary_lines] = superpixel(I, Level)
    %L, a label matrix of type double, and 
    %NumLabels, the actual number of superpixels that were computed.
    [L,N] = superpixels(I,Level);
    %obtain the boundary lines as a mask
    boundary_lines = boundarymask(L);
    %display the boundary lines overlayed on the image
    %figure('NumberTitle', 'off', 'Name', 'Overlay Segmented Boundaries'), imshow(imoverlay(I,boundary_lines,'white'),'InitialMagnification',67);***
    %setup the segmented output variable
    segmented = zeros(size(I),'like',I);
    %obtain the linear indices of the labels
    idx = label2idx(L);
    %obtain the rows and columns to convert the linear indices to 2D
    numRows = size(I,1);
    numCols = size(I,2);
    %loop through all the labelled sections
    for labelVal = 1:N
        %obtain the 2D indices for each colour layer's segment
        redIdx = idx{labelVal};
        greenIdx = idx{labelVal}+numRows*numCols;
        blueIdx = idx{labelVal}+2*numRows*numCols;
        %assign mean values to the segments for colouring
        segmented(redIdx) = mean(I(redIdx));
        segmented(greenIdx) = mean(I(greenIdx));
        segmented(blueIdx) = mean(I(blueIdx));
    end  
end

%Function that checks if the current index is out of bounds
%input:
%rx: width of image
%gy: height of image
%i: kernel index width position
%J: kernel index height position
%output: true or false depnding on boundary condition
function output = check(rx, gy , i, j)
    %Sets default value
    output = false;
    %Returns false if the value is out of bounds
    if (i>0) && (i<rx) && (j>0) && (j<gy) 
        output = true;
    end

end

%convolution function
%parameters:
%I: the original image
%B: the boundary image
%h: the filter
%return:
%output: the filtered image
function output = selective_conv(I,B,h)
    %create a copy of the image
    I_copy = I;
    %obtain the filter dimensions
    [m,n] = size(h);
    %obtain the number of columns and rows to pad the image by
    pad_m = floor(double(m)/double(2)); 
    pad_n = floor(double(n)/double(2)); 
    %pad the image
    I_copy = padarray(I_copy,[pad_m pad_n],'symmetric','both');
    %create output image
    [row,col] = size(I);
    %setup the output image
    output = uint8(zeros(row,col));
    %obtain an array of positions where the boundary exists
    k = find(B); %gets linear index where B is non-zero*NB, goes column-wise
    [last_k,~] = size(k);
    %obtain width and height of B to covert linear index to 2D
    [bx,by] = size(B);

    %at each k point, apply convolution for that neighbourhood to obtain the output sum    
    %loop through k
    for i = 1:1:last_k
        sum = 0;
        %obtain the linear index
        lin_idx = k(i,1);
        %obtain the 2D index
        [x,y] = ind2sub([bx by],lin_idx);
        %obtain the neighbourhood
        neighbourhood = single(I_copy(x:x+2*pad_m,y:y+2*pad_n));
        %loop through the pixels in the filter
        for a = 1:1:m
            for b = 1:1:n
                %sum the values multiplied element-wise of the filter and image in the neighbourhood 
                sum = sum + h(a,b)*neighbourhood(a,b);
            end
        end
	%set the output at the position to the convoluted value
        output(x,y) = uint8(sum);
    end
end
