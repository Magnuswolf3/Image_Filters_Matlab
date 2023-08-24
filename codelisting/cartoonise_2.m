%Applying Bilateral Filter to homogenise colour regions
smooth = bilateralFilter(I, downsampleRate, bilKernSize, bilTimes);
%Applying Median Filter to smoothen small/remaining artifacts
smooth = medfilt3(smooth, [7 7 1]);
%Quantizing the Colours
quant = quantizeColour(smooth,quantizeFactor);
%Converting edges to uint8
large_edge_regions = uint8(large_edge_regions*255); 
large_edge_regions_less = uint8(large_edge_regions_less*255); 
%Combines the different layers of the image to produce an RGB image
final = cat(3, uint8(layerR), uint8(layerG), uint8(layerB));
final_less = swap_edge(final, large_edge_regions_less);
%Overlays edges onto quantised image
final = swap_edge(final, large_edge_regions);

%Overlays edges onto image
%input:
%final: a processed RGB image
%leRegions: and image of black edges
%output: returns an image with the edges swapped out
function output = swap_edge(final, leRegions)
    %Swaps out colour of image where edges reside
    output = final -leRegions;
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
