%Standard Deviation Function
%input: 
%I: original RGB image
%x,y: contain positional information
%n: size of the neighbourhood used for the image
%vf: variance factor variable that is multiplied to the borders to change their intensities
%output: the standard deviation of the current neighbourhood
function output = std(I,x,y,n,vf)
    %obtain neighbourhood for current filter origin position
    neighbourhood = I(x-n:x+n,y-n:y+n);   
    %Calculating the variance of current neighbourhood 
    variance = vf * var(neighbourhood(:));  
    %Calculating the standard deviation from the variance
    deviation = sqrt(variance);    
    %Returning the standard deviation as the result for the current pixel
    output = deviation;
end
