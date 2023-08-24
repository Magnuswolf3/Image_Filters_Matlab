%Team: Dhannya Mathew - 1706542 & Saatwik Kambadkone - 1601869
%file name
source_file = "1.jpg";
number = split(source_file,".");
input_str = "images/";
input_path = input_str + source_file;
output_str = "output/lumin_edge/" + number(1) + "_lumin_edge";
output_path = output_str + "." + number(2);
%Read in image
I= imread(input_path);

%Size of filter, determines thickness of the edge
n = 1;
%Size of original image
[M, N, D] = size(I);
%Variance Factor, changes brightness of edge
vFactor = 5;
%Gaussian Filter, changes how focussed the blur will be
gFilter = 10;
%Glow Factor, changes how bright the glow will be
gFactor = 0.3;

%Splitting up the RGB channels
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

%Padding the image with mirror padding to get rid of artifacts on edges
R = padarray(R, [n n], 'symmetric', 'both');
G = padarray(G, [n n], 'symmetric', 'both');
B = padarray(B, [n n], 'symmetric', 'both');

%Convert Images to Double
R = im2double(R);
G = im2double(G);
B = im2double(B);

%Call the Standard Deviation Edge Detection Method
ROut = SDED(R, n, vFactor);
GOut = SDED(G, n, vFactor);
BOut = SDED(B, n, vFactor);

%Unpadded the Image
ROut = ROut(n+1: M+n, n+1: N+n);
GOut = GOut(n+1: M+n, n+1: N+n);
BOut = BOut(n+1: M+n, n+1: N+n);

%Putting all the channels back together
final_1 = cat(3, ROut, GOut, BOut);

%Converting to Linear Space and Applying Gaussian Filter in Linear Space
f_lin = rgb2lin(final_1);
f_lin = imgaussfilt(f_lin, gFilter);

%Calculating the Glow that should be added to the Image
f_lin = glowCalculate(final_1, f_lin, gFactor);

%Adding the Linear Space to the Original Image
final_2 = final_1 + f_lin;

%Displaying Results
figure('NumberTitle', 'off', 'Name', 'Lumin Edge Process'), subplot(3,1,1),imshow(I),title('Original');
subplot(3,1,2), imshow(final_1),title('Edges');
subplot(3,1,3), imshow(f_lin),title('Glow');
figure('NumberTitle', 'off', 'Name', 'Lumin Edge Result'), imshow(final_2);
%write out result
imwrite(final_2, output_path);

%Calculate the amount of glow that will be applied to the image based off
%the contrast in the image itself
%input:
%I: An image containing the edges of another RGB image
%f_lin: The linear space of the image
%gf: Glow factor variable that will be applied to image
%output: A glow that will be added to the output later
function output = glowCalculate(I, f_lin, gf)
    %Get standard deviation of image
    standard = std2(I);
    %Work out a factor to multiply the glow by based off standard deviation
    mult = gf/(standard);
    %multiply the glow component by the calculated factor
    output = mult * f_lin;
end


%Main function of the program
%input:
%I: an RGB image
%n: size of the neighbourhood used for the image
%vf: variance factor variable that is multiplied to the borders to change
%their intensities
%output: Returns an image with the edges calculated using standard deviation
function output = SDED(I, n, vf)
    %Size parameters of Padded Image
    [M,N] = size(I);
    %Stores output Image
    temp = zeros(M,N);
    
    %Iterate through every pixel
    for l1 = 1+n:1:M-n
        for l2 = 1+n:1:N-n
            %Figure out standard deviation of a neighbourhood for each pixel
            temp(l1,l2) = std(I,l1,l2,n, vf);
        end
    end
    %Return the result
    output = temp;
end

%Standard Deviation Function
%input: 
%I: original RGB image
%x,y: contain positional information
%n: size of the neighbourhood used for the image
%vf: variance factor variable that is multiplied to the borders to change
%their intensities
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
