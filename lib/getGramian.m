function [ A ] = getGramian (Factor_Matrices, Active_Columns, GPU_Computing)
%getGramian v1.0
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%Obtain a Gram matrix by selecting a subset of columns from a
%kronecker matrix given by kronecker product of matrices in the Kron_Cell_Array

%% Function Call
%[ X ] = getGramian(Factor_Matrices, Active_Columns, GPU_Computing);

%% Inputs 

%Variable        Type               Description

%Factor_Matrices (Cell Array)   = Contains the 1 dimensional matrices of the separable kronecker matrix 
%Active_Columns  (Numeric Array)= Array containing indices of selected columns
%GPU_Computing   (Logical)      = True/False : If True run on GPU

%% Outputs  

%A (Matrix) = Resultant kronecker sub matrix

%% getKroneckerSubMatrix

A = zeros(length(Active_Columns));

kronMatrixCount = length(Factor_Matrices);
dim_array = zeros(1,kronMatrixCount);
for n = 1:kronMatrixCount
    dim_array(n) = size(Factor_Matrices{n},2);
end

parfor i = 1:length(Active_Columns)
    
    new_column_index = Active_Columns(i);

    factor_column_indices = getKroneckerFactorColumnIndices( kronMatrixCount, new_column_index, dim_array );
    g_k = getKroneckerMatrixColumn( Factor_Matrices, factor_column_indices, GPU_Computing );  
    gramian_row = g_k(Active_Columns);
    A(i,:) = gather(gramian_row);
end
end

