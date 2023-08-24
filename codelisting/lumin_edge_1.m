%Main function of the program
%input:
%I: an RGB image
%n: size of the neighbourhood used for the image
%vf: variance factor variable that is multiplied to the borders to change their intensities
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

