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
