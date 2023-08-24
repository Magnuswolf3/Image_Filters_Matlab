%remove noise using median filter
RGB = medfilt3(I,[7 7 1]);
subplot(3,2,2),imshow(RGB),title('Median Filtered');
%convert image to YCbCr space
gray = rgb2ycbcr(RGB);
%obtain the lumin field
layerY = squeeze(gray(:,:,1));
%apply inbuilt canny edge detection
[edge_Y, large_edge_regions,large_edge_regions_less] = edge_detect(layerY);

%Find the edges of an image then dilates to make them larger
%input:
%layerY: the Y layer of an RGB image
%output: returns the edges of an RG%remove noise using median filter
RGB = medfilt3(I,[7 7 1]);
%convert image to YCbCr space
gray = rgb2ycbcr(RGB);
%obtain the lumin field
layerY = squeeze(gray(:,:,1));
%apply inbuilt canny edge detection
[edge_Y, large_edge_regions,large_edge_regions_less] = edge_detect(layerY);

%Does mathematical morphological operations on the edge image
%input:
%layerY: the Y layer of an RGB image
%output: returns the edges of an RGB image's Y layer
function [edge_Y,large_edge_regions, large_edge_regions_less]= edge_detect(layerY)
    canny = edge(layerY,'canny',([]),'zerocross'); 
    %dilate the one pixel edges
    cross = strel([0 1 0; 1 1 1; 0 1 0]);
    %thickens edges
    edge_Y = imdilate(canny,cross); 
    %Using morphology to reduce the number of stray, pointless edges
    large_edge_regions = bwmorph(edge_Y, 'skel');

    large_edge_regions = bwmorph(large_edge_regions, 'shrink');
    large_edge_regions = bwmorph(large_edge_regions, 'hbreak');
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
endB image's Y layer
function [edge_Y,large_edge_regions, large_edge_regions_less]= edge_detect(layerY)
    canny = edge(layerY,'canny',([]),'zerocross'); 
    %dilate the one pixel edges
    cross = strel([0 1 0; 1 1 1; 0 1 0]);
    %thickens edges
    edge_Y = imdilate(canny,cross); 
    %Using morphology to reduce the number of stray, pointless edges
    large_edge_regions = bwmorph(edge_Y, 'skel');

    large_edge_regions = bwmorph(large_edge_regions, 'shrink');
    large_edge_regions = bwmorph(large_edge_regions, 'hbreak');
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
end
