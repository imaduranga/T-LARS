function [  ] = show_tlars( Ax, Y, Tensor_Norm, Norm_R, Tensor_Dimensions )
%show_tlars v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%This function display tlars results

%% Function Call

% show_tlars( Ax, Y, Tensor_Norm, Norm_R, Dim_Array );

%% Inputs 

%Variable        Type               Description

%Ax            (Numeric Array)  = Current result in a vector form
%Y             (N-D Array)      = Normalizerd input data tensor
%Tensor_Norm   (Numeric)        = Norm of the input data tensor
%Dim_Array     (Numeric Array)  = Dimensions of the core tensor as an array

%% show_tlars 

figureNumber = 2;
draw_image = false;

Y = double(Y);


if length(Tensor_Dimensions) > 1

    Irn = reshape(Ax,Tensor_Dimensions);
    
    if length(Tensor_Dimensions) == 2
        Ir= Irn.*Tensor_Norm;
        Iry = Y.*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 3
        Ir= Irn(:,:,5).*Tensor_Norm;
        Iry = Y(:,:,5).*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 4
        Ir= Irn(:,:,5,5).*Tensor_Norm;
        Iry = Y(:,:,5,5).*Tensor_Norm;
        draw_image = true;
    elseif length(Tensor_Dimensions) == 5
        Ir= Irn(:,:,5,5,5).*Tensor_Norm;
        Iry = Y(:,:,5,5,5).*Tensor_Norm;
        draw_image = true;
    end

end

if length(Norm_R) <= 5
   figure(figureNumber);
end

fig = gcf;

if fig.Number ~= figureNumber
    r = groot;
    for i = 1:length(r.Children)    
        if r.Children(i).Number == figureNumber
           fig = r.Children(i);
        end  
    end
    set(0,'CurrentFigure',fig);
end

if draw_image
    subplot(1,3,1);
    imshow(mat2gray(double(full(Iry))));
    title('Original','Interpreter','latex');
    subplot(1,3,2);
    imshow(mat2gray(double(full(Ir))));
    title('Reconstructed','Interpreter','latex');
    subplot(1,3,3);
end

plot(Norm_R);
title('$||R||_2$ Vs. \# of Iterations','Interpreter','latex');
xlabel('\# of Iterations','Interpreter','latex') % x-axis label
ylabel('$||R||_2$','Interpreter','latex') % y-axis label
drawnow
end

