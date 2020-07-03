function [ y ] = kroneckerMatrixPartialVectorProduct( Factor_Matrices, Active_Columns, x, Use_Transpose, GPU_Computing )
%kroneckerMatrixPartialVectorProduct v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%kroneckerMatrixPartialVectorProduct function calculates the product
%between a matrix A and a vector x (y = Ax) using full multilinear product. 
%Columns of matrix A can be obtained by kronecker product of respective columns in the kron_cell_array. 
%If matrix B is the kronecker product of all factor matrices in Factor_Matrices, columns of A is a subset of columns of B.
%If transpose = True, y = A'x is calculated

%% Function Call

%[ y ] = kroneckerMatrixPartialVectorProduct( Factor_Matrices, Active_Columns, x, Use_Transpose, GPU_Computing );

%% Inputs 

%Variable        Type           Description

%Factor_Matrices (Cell Array)   = Contains the 1 dimensional matrices of the separable kronecker matrix A 
%Active_Columns  (Numeric Array)= Array containing Active elements of the matrix A to be used in matrix vector calclation
%x               (Numeric Array)= Vector to be multiplied with the matrix A
%Use_Transpose   (Logical)      = Obtain a kronecker product of transposed factor matrices
%GPU_Computing   (Logical)      = True/False : If True run on GPU


%% Outputs  

%y (Numeric Array) = Result vector of y = Ax; or y = A'x


%% kroneckerMatrixPartialVectorProduct

    if isempty(Use_Transpose) 
       Use_Transpose = false;
    end

    % Construct dimension array
    kronMatrixCount = length(Factor_Matrices);    
    dim_array = zeros(1,kronMatrixCount);

    if Use_Transpose
        for n = 1:kronMatrixCount
            dim_array(n) = size(Factor_Matrices{n},1);
        end             
    else
        for n = 1:kronMatrixCount            
            dim_array(n) = size(Factor_Matrices{n},2);
        end
    end
    
    %Set vector vx to have x as only non-zero elements
    column_length = prod(dim_array);
    
    if GPU_Computing
        vx = zeros(1, column_length ,'gpuArray');
    else
        vx = zeros(1, column_length);
    end    
    
    if isrow(x)
        vx(Active_Columns) = x;
    else
        vx(Active_Columns) = x';
    end
    
    % Reshape vx to a tensor format
    if kronMatrixCount > 1
        X = reshape(vx,dim_array);   
    else
        X = vx';
    end
   
    %Obtain the full multilinear product of the tensor X and factor matrices  
    Y = fullMultilinearProduct( X, Factor_Matrices, Use_Transpose, GPU_Computing );    
    y = vec(Y);
end

