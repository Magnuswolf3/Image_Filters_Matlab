%Calculate the amount of glow that will be applied to the image based off the contrast in the image itself
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
