%https://softwarebydefault.com/2013/06/29/oil-painting-cartoon-filter/
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
    %intensity scaling the intensities in the original image to match chosen level
    %puts it in the range of 0 to Level - gray now holds the intensity bin indices
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
            %compute the indices for the output image with respct to the loop variables
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
