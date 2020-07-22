function [ dI, GInv] = getDirectionVector( GInv, zI, Gramian_Cell_Array, Active_Columns, Add_Column_Flag, Changed_Column_Index, Changed_Active_Column_Index, Tensor_Dimensions, GPU_Computing ) 
%getDirectionVector v1.1
%Author : Ishan Wickramsingha
%Date : 2019/10/31

%getDirectionVector function update GInv for column addition and removal
%using the shur complements inversion formula for column addition and removal.
%Finally calculate the direction vector dI from the updated GInv.
%This function support GPU computation and if the GPU memory is limited the
%function will use less GPU memory for computations.

%% References

% If you use this code in a scientific publication, please cite the following paper:

% Wickramasingha I, Elrewainy A, Sobhy M, Sherif SS. Tensor Least Angle Regression for Sparse Representations of Multidimensional Signals. Neural Comput. 2020;32(9):1-36. doi:10.1162/neco_a_01304

%% Function Call
%[ dI, GInv] = getDirectionVector( GInv, zI, Gramian_Cell_Array, Active_Columns, Add_Column_Flag, Changed_Column_Index, Changed_Active_Column_Index, Tensor_Dimensions, GPU_Computing );

%% Inputs 

%Variable        Type           Description

%GInv                       (Matrix)        = Inverse of the Gramian
%zI                         (Numeric Array) = Vector to multiply with GInv
%Gramian_Cell_Array         (Cell Array)    = Contains the 1 dimensional matrices of the separable kronecker Gram matrix of the dictionry: G = D'D 
%Active_Columns             (Numeric Array) = Array containing Active elements of the matrix A to be used in matrix vector calclation
%Add_Column_Flag            (Logical)       = If true a column is added, else a column is removed from the active set          
%Changed_Column_Index       (Numeric)       = Index of the changed column in the separable dictionary D
%Changed_Active_Column_Index(Numeric)       = Index of the changed column in the Active Columns Array
%Tensor_Dimensions          (Logical)       = Dimensions of the tensor as an array
%GPU_Computing              (Logical)       = True/False : If True run on GPU


%% Outputs  

%dI   (Numeric Array) = Calculated direction vector : dI = (A'A)\zI
%GInv (Matrix)        = Updated inverse of the Gramian


%% Initialization

N = length(Active_Columns);
order = length(Gramian_Cell_Array);

if size(GInv,2) <= N+1        
    GInv(N+100,N+100) = 0;
end

%% GPU Computing parameters
if GPU_Computing    
    limitedGPU = false;
    minimumGPU = false;
    
    gpu = gpuDevice();
    total_gpu_memory = gpu.TotalMemory;
    free_gpu_memory = gpu.FreeMemory;    
    
    element_count = N*N;
    min_gpu_memory_required = min(5e8, 3*element_count*8) ; %500MB
    required_gpu_memory = min_gpu_memory_required + element_count*8 + ~isa(GInv,'gpuArray')*element_count*8;
    required_limited_gpu_memory = min_gpu_memory_required + 3*element_count*8 + ~isa(GInv,'gpuArray')*element_count*8;
    
    max_gpu_element_count = round(0.7*total_gpu_memory/8);
    max_gpu_process_elemet_count = 0.1*free_gpu_memory/8;
    
    if free_gpu_memory < min_gpu_memory_required 
        fprintf('GPU Computing Disabled. Available GPU Memory = %g \n',free_gpu_memory);
        GPU_Computing = false;        
    elseif max_gpu_element_count < element_count || free_gpu_memory < required_gpu_memory
        fprintf('Minimum GPU Function. Available GPU Memory = %g \n',free_gpu_memory);
        minimumGPU = true;
    elseif free_gpu_memory < required_limited_gpu_memory
        fprintf('Limited GPU Function. Available GPU Memory = %g \n',free_gpu_memory);        
        limitedGPU = true;
    end
    
    if ~GPU_Computing || minimumGPU
        if isa(GInv,'gpuArray') 
            GInv = gather(GInv);
            dI = zeros(N,1);
        end
    elseif ~isa(GInv,'gpuArray')
        GInv = gpuArray(full(GInv));
        dI = gpuArray(zeros(N,1));          
    end

end

%% Update GInv and Calculate direction Vector dI

if  Add_Column_Flag == 1    
%Add a column
    
    % Calculate update paramenters for the added row
    factor_column_indices = getKroneckerFactorColumnIndices( order, Changed_Column_Index, Tensor_Dimensions );
    g_k = getKroneckerMatrixColumn( Gramian_Cell_Array, factor_column_indices, GPU_Computing ); 
    g_a = g_k(Active_Columns);
    clear g_k;
    
    if length(g_a) > 1

        b = ones(N,1);
        
        if GPU_Computing
            b = gpuArray(b);
            g_a = full(g_a);
        end
        
        % For limited or minimum GPU usage run this code
        if GPU_Computing && ( minimumGPU || limitedGPU )    
            
            process_matrix_element_count = min(max_gpu_process_elemet_count,round(0.1*gpu.FreeMemory/8));            
            stepSize = ceil(process_matrix_element_count/(N-1));
            steps = min(N-1,ceil((N-1)/stepSize));
            
            startIndex = 1;
            endIndex = min(N-1,stepSize);
            
            for i = 1:steps
                b(startIndex:endIndex) = -GInv(startIndex:endIndex,1:N-1)*g_a(1:N-1);
                startIndex = startIndex + stepSize;
                endIndex = min(N-1,endIndex + stepSize);
            end  
            
        % For unlimited GPU usage run this code
        else
            b(1:N-1) = -GInv(1:N-1,1:N-1)*g_a(1:N-1);
        end        
                
        alpha = 1/(g_a(N) + g_a(1:N-1)'*b(1:N-1));
        clear g_a;
               
        % Update GInv and dI
        % For limited or minimum GPU usage run this code
        if GPU_Computing && ( minimumGPU || limitedGPU )  
            
            process_matrix_element_count = min(max_gpu_process_elemet_count,round(0.1*gpu.FreeMemory/8));

            if minimumGPU                
                alpha = gather(alpha);            
            end                        

            stepSize = ceil(process_matrix_element_count/N);
            steps = min(N,ceil(N/stepSize));
            startIndex = 1;
            endIndex = min(N,stepSize);

            for i = 1:steps            
                if minimumGPU
                    GIR = gpuArray(GInv(startIndex:endIndex,1:N));
                    GIR = GIR + alpha*(b(startIndex:endIndex)*b');
                    dI(startIndex:endIndex) = gather(GIR*zI);
                    GInv(startIndex:endIndex,1:N) = gather(GIR);
                else
                    GInv(startIndex:endIndex,1:N) = GInv(startIndex:endIndex,1:N) + alpha*(b(startIndex:endIndex)*b');
                    dI(startIndex:endIndex) = GInv(startIndex:endIndex,1:N)*zI;
                end            

                startIndex = startIndex + stepSize;
                endIndex = min(N,endIndex + stepSize);
            end
            
        % For unlimited GPU usage run this code
        else
            GInv(1:N,1:N) = GInv(1:N,1:N) + alpha*(b*b');
            dI = GInv(1:N,1:N)*zI;
        end        
        clear b;
    else
        GInv(1,1) = 1/g_a;
        dI = GInv(1,1)*zI;
    end   
    
%Calcuations for removing a column    
elseif Add_Column_Flag == 0    
%Reorder the GInv for the removed column to be the last column
%Finv,ab and alpha are calculated assuming the reordered column structure
        
    alpha = GInv(Changed_Active_Column_Index,Changed_Active_Column_Index);   
    ab = GInv(1:N+1,Changed_Active_Column_Index);
    ab(Changed_Active_Column_Index) = [];    
   
    % For limited or minimum GPU usage run this code
    if GPU_Computing && ( minimumGPU || limitedGPU )  
                
        if minimumGPU            
            alpha = gpuArray(alpha);
            ab = gpuArray(ab);        
        end
        
        %Shift GInv to remove removed column
        %Calculate step size for use in GPU without memory exception
        process_matrix_element_count = min(max_gpu_process_elemet_count,round(0.1*gpu.FreeMemory/8));
        stepSize = ceil(process_matrix_element_count/(N+1));
        steps = min(N+1-Changed_Active_Column_Index,ceil((N+1-Changed_Active_Column_Index)/stepSize));

        startIndex = Changed_Active_Column_Index;
        endIndex = min(N+1,Changed_Active_Column_Index+stepSize);        
        
        for i = 1:steps
            GInv(:,startIndex:endIndex) = GInv(:,startIndex+1:endIndex+1);
            GInv(startIndex:endIndex,:) = GInv(startIndex+1:endIndex+1,:); 
            startIndex = startIndex + stepSize;
            endIndex = min(N+1,endIndex + stepSize);
        end        

        %Calculate GInv and dI
        steps = min(N,ceil(N/stepSize));
        startIndex = 1;
        endIndex = min(N,stepSize);   

        for i = 1:steps

            if minimumGPU
                GIR = gpuArray(GInv(startIndex:endIndex,1:N));
                GIR = GIR - ab(startIndex:endIndex)*ab'/alpha;
                dI(startIndex:endIndex) = gather(GIR*zI);
                GInv(startIndex:endIndex,1:N) = gather(GIR);
            else
                GInv(startIndex:endIndex,1:N) = GInv(startIndex:endIndex,1:N) - ab(startIndex:endIndex)*ab'/alpha;
                dI(startIndex:endIndex) = GInv(startIndex:endIndex,1:N)*zI;
            end            

            startIndex = startIndex + stepSize;
            endIndex = min(N,endIndex + stepSize);       

        end
        
    % For unlimited GPU usage run this code
    else
        %Shift GInv to remove removed column
        GInv(:,Changed_Active_Column_Index:N+1) = GInv(:,Changed_Active_Column_Index+1:N+2);
        GInv(Changed_Active_Column_Index:N+1,:) = GInv(Changed_Active_Column_Index+1:N+2,:);

        %Calculate GInv and dI
        GInv(1:N,1:N) = GInv(1:N,1:N) - (ab*ab')/alpha; 
        dI = GInv(1:N,1:N)*zI;
    end
    
    clear ab;
    
%If neither a column is added or remved run this code    
else
    dI = GInv(1:N,1:N)*zI;
end

%Make sure dI is a column vector
if isrow(dI)
    dI = dI';
end

end


