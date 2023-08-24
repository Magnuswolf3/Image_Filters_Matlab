%Team: Dhannya Mathew - 1706542 & Saatwik Kambadkone - 1601869
%file name
source_file = "5.jpg";
number = split(source_file,".");
input_str = "images/";
input_path = input_str + source_file;
output_str_1 = "output/cartoonise/" + number(1) + "_cartoonised_thickedge";
output_str_2 = "output/cartoonise/" + number(1) + "_cartoonised_thinedge";
output_path_1 = output_str_1 + "." + number(2);
output_path_2 = output_str_2 + "." + number(2);
%Read in the image
I = imread(input_path);
figure('NumberTitle', 'off', 'Name', 'Cartoonise Process'),subplot(3,2,1),imshow(I),title('Original');
%----------------------------------------------------------------------
%remove noise using median filter 
RGB = medfilt3(I,[7 7 1]);
subplot(3,2,2),imshow(RGB),title('Median Filtered');
%convert image to YCbCr space
gray = rgb2ycbcr(RGB);
%obtain the lumin field
layerY = squeeze(gray(:,:,1));
subplot(3,2,3),imshow(layerY),title('Lumin Layer');

%gray = rgb2gray(RGB);

%apply edge detection (including morphological techniques)
H1 = fspecial('gaussian',7,3); %hsize, sigma
H2 = fspecial('gaussian',7,7);
DoG = H1 - H2;
edge = conv2(layerY,DoG,'same');
edge = imbinarize(edge,0);
%%subplot(3,2,4), imshow(edge),title('Edge DOG');

%apply inbuilt edge detection
[edge_Y, large_edge_regions,large_edge_regions_less] = edge_detect(layerY);

%display the edges
subplot(3,2,4), imshow(edge_Y),title('Canny Edge');

%display the edge regions that succeeded the filtering
subplot(3,2,5), imshow(large_edge_regions),title('Large Edge Regions');
%----------------------------------------------------------------------
%Colour Smoothing
%Size of Bilateral Kernal Size (nxn)
bilKernSize = 9;
%Number of times Bilateral Filter is Applied
bilTimes = 14;
%The Downsampling Rate Used for the Image
downsampleRate = 2;
%Picking the Factor for Quantization of Colours
quantizeFactor = 24;

%Applying Bilateral Filter to homogenise colour regions
smooth = bilateralFilter(I, downsampleRate, bilKernSize, bilTimes);

%Applying Median Filter to smoothen small/remaining artifacts
smooth = medfilt3(smooth, [7 7 1]);

%Quantizing the Colours
quant = quantizeColour(smooth,quantizeFactor);

%Plot quantised layers
subplot(3,2,6), imshow(quant),title('Quantized Colour');

%Converting edges to uint8
large_edge_regions = uint8(large_edge_regions*255); 
large_edge_regions_less = uint8(large_edge_regions_less*255); 

%----------------------------------------------------------------------
%Splitting quantised layers
layerR = quant(:,:,1); 
layerG = quant(:,:,2);
layerB = quant(:,:,3);

%Getting the mean of the image in order to automate colour of edges
meanR = mean(layerR(:));
meanG = mean(layerG(:));
meanB = mean(layerB(:));

%Makes sure the colour of the mean is only applied to the edges
edgeR  = bitand(large_edge_regions, uint8(meanR));
edgeG  = bitand(large_edge_regions, uint8(meanG));
edgeB  = bitand(large_edge_regions, uint8(meanB));
coloredEdges = cat(3, uint8(edgeR), uint8(edgeG), uint8(edgeB));

%Makes the colour darker than the mean
coloredEdges = brighten(double(coloredEdges), -.8);
coloredEdges = uint8(coloredEdges*255);

%Combines the different layers of the image to produce an RGB image
final = cat(3, uint8(layerR), uint8(layerG), uint8(layerB));
final_less = swap_edge(final, large_edge_regions_less);
final = swap_edge(final, large_edge_regions);
%Swaps the edges of the image

%figure('NumberTitle', 'off', 'Name', 'Cartoonised'), imshow(coloredEdges);
figure('NumberTitle', 'off', 'Name', 'DOG Edge Detection'), imshow(edge); 
figure('NumberTitle', 'off', 'Name', 'Cartoonised Thin and Thick Edges'), imshow(final_less); 
figure('NumberTitle', 'off', 'Name', 'Cartoonised Thick Edges'), imshow(final); 
%write out result
imwrite(final, output_path_1);
imwrite(final_less, output_path_2);
%figure('NumberTitle', 'off', 'Name', 'Cartoonised'), imshow(imoverlay(quant,large_edge_regions,'black'),'InitialMagnification',67);
%----------------------------------------------------------------------

%Overlays edges onto image
%input:
%final: a processed RGB image
%leRegions: and image of black edges
%output: returns an image with the edges swapped out
function output = swap_edge(final, leRegions)
    %Swaps out colour of image where edges reside
    output = final -leRegions;
end

%Does mathematical morphological operations on the edge image
%input:
%layerY: the Y layer of an RGB image
%output: returns the edges of an RGB image's Y layer
function [edge_Y,large_edge_regions, large_edge_regions_less]= edge_detect(layerY)
    canny = edge(layerY,'canny',([]),'zerocross'); %[low_thresh high_thresh]
    %dilate the one pixel edges
    cross = strel([0 1 0; 1 1 1; 0 1 0]);
    %disk = strel('disk',2,0);
    edge_Y = imdilate(canny,cross); %thickens edges
    %Using morphology to reduce the number of stray, pointless edges
    large_edge_regions = bwmorph(edge_Y, 'skel');

    %large_edge_regions = large_edge_regions - bwmorph(large_edge_regions, 'endpoints');
    large_edge_regions = bwmorph(large_edge_regions, 'shrink');
    %large_edge_regions = bwmorph(large_edge_regions, 'clean');
    %large_edge_regions = large_edge_regions - bwmorph(large_edge_regions, 'endpoints');
    large_edge_regions = bwmorph(large_edge_regions, 'hbreak');
    %large_edge_regions = bwmorph(large_edge_regions, 'spur');
    large_edge_regions = large_edge_regions - bwmorph(large_edge_regions, 'endpoints');
    large_edge_regions = bwareaopen(large_edge_regions, 50);
    large_edge_regions = bwmorph(large_edge_regions, 'diag');
    large_edge_regions = bwmorph(large_edge_regions, 'thicken',2);
    large_edge_regions = bwmorph(large_edge_regions, 'open');
    
    %thick and thin edges
    large_edge_regions_temp = bwmorph(large_edge_regions, 'skel', inf);
    large_edge_regions_temp = bwmorph(large_edge_regions, 'skel') - bwareaopen(large_edge_regions_temp, 100);
    large_edge_regions_temp = bwmorph(large_edge_regions_temp, 'diag');
    large_edge_regions_less = bwmorph(large_edge_regions_temp, 'thicken', 0.5);
    large_edge_regions_temp = bwmorph(large_edge_regions, 'skel', inf);
    large_edge_regions_temp = bwareaopen(large_edge_regions_temp, 100);
    large_edge_regions_temp = bwmorph(large_edge_regions_temp, 'diag');
    large_edge_regions_temp = bwmorph(large_edge_regions_temp, 'thicken', 1.5);
    large_edge_regions_less = large_edge_regions_less + large_edge_regions_temp;
    large_edge_regions_less = bwmorph(large_edge_regions_less, 'open');
    large_edge_regions_less = bwareaopen(large_edge_regions_less, 50);
    %large_edge_regions_less = large_edge_regions_less + large_edge_regions;
    
    %Weird geometric filter
    %large_edge_regions = bwmorph(large_edge_regions, 'thicken', 800);
    
end

%Applies the bilateral filter to an image
%input:
%I: an RGB image
%ds: the downsampling rate that will be used for this image
%bks: kernel size that will be used for bilateral filters
%bt: number of times the bilateral filter will be applied
%output: An RGB image that has been smoothed using the bilateral filter
function output = bilateralFilter(I, ds, bks, bt)
    %Downsampling to make bilateral filter less expensive
    [M,N,~] = size(I);
    I = I(1:ds:end,1:ds:end,:);
    
    %Applying Bilateral Filter Over Several Iterations to Smooth Image
    for l1 = 1:1:bt
        I = imbilatfilt(I,'NeighborhoodSize', bks);
    end
    %Upsampling to restore to original size
    I = imresize(I,[M N]);
    %steps = double(1/ds);
    %I = I(1:steps:end,1:steps:end,:);
    output = I;
end

%Decreases the number of colours in an image
%input:
%I: an RGB image
%a: the factor that we're gonna use to reduce the number of images by
%output: the RGB image with it's colours reduced
function output = quantizeColour(I, a)
    [M,N,~] = size(I);
    
    %Splitting up the Channels to Apply Filter to Each Channel
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    
    %Decreasing the Number of Colours in each Channel
    for l1 = 1:1:M
        for l2  = 1:1:N
            R(l1, l2) = floor(R(l1, l2)/a) * a;
            G(l1, l2) = floor(G(l1, l2)/a) * a;
            B(l1, l2) = floor(B(l1, l2)/a) * a;
        end
    end
    
    %Joining the 3 Channels Back Together
    output = cat(3, R, G, B);
end