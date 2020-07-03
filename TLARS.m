function varargout = TLARS( Y, D_Cell_Array, Active_Columns_Limit, varargin )
%TLARS v1.0.0
%Authors : Ishan Wickramasingha, Ahmed Elrewainy, Michael Sobhy, Sherif S. Sherif
%Date : 2019/10/31

%MATLAB Version : MATLAB R2017b and above

%% References

% If you use this code in a scientific publication, please cite following papers:

% Wickramasingha, Ishan, Ahmed Elrewainy, Michael Sobhy, and Sherif S. Sherif. " Tensor Least Angle Regression for Sparse Representations of Multi-dimensional Signals." Neural computation (2020) (Accepted) 
% Elrewainy, A., & Sherif, S. S. (2019). Kronecker least angle regression for unsupervised unmixing of hyperspectral imaging data. Signal, Image and Video Processing, 14(2), 359–367. https://doi.org/10.1007/s11760-019-01562-w

%% Function Call
%[ X ] = TLARS( Y, D_cell_array, Active_Columns_Limit);
%[ X, Active_Columns ] = TLARS( Y, D_cell_array, Active_Columns_Limit, ...);
%[ X, Active_Columns, x ] = TLARS( Y, D_cell_array, Active_Columns_Limit, ...);
%[ X, Active_Columns, x, Parameters, Stat ] = TLARS( Y, D_Cell_Array, Active_Columns_Limit, Tolerence, X, L0_Mode, GPU_Computing, Plot, Debug_Mode, Path, Iterations, Precision_factor );

%% Inputs 
%Variable        Type       Default   Description
%Y             (N-D Array)          = Input data tensor
%D_cell_array  (Cell Array)         = Contains the 1 dimensional dictionarary matrices of the separable dictionary D
%Active_Columns_Limit(Numeric)1e+6  = Depends on the required sparsity
%Tolerence     (Numeric)    0.001   = The norm of the residual error as a tolerence to stop the algorithm
%X             (N-D Array)          = Previous partially calculated solution. If X = 0 TLARS runs from the begining.
%L0_mode       (Logical)    False   = True/False : True for L0 or false for L1 Minimization
%GPU_Computing (Logical)    False   = True/False : If True run on GPU if available
%Plot          (Logical)    False   = Plot norm(r) at runtime
%Debug_Mode    (Logical)    False   = True/False : Save TLARS variable into a .mat file given in path in debug mode
%Path          (string/char) ''     = Path to save all variables in debug mode
%Iterations    (Numeric)    numel(Y)= Maximum Number of iteratons to run
%Precision_factor (Numeric)  5      = Round to 5 times the machine precission 

%% Outputs  
%
%X              (N-D Array)  = Output coefficients in a Tensor format 
%Active_Columns (Numeric Array)  = Active columns of the dictionary
%x              (Numeric Array)= Output coefficients for active columns in a vector format
%
%Parameters = Algorithm Parameters class
%     Parameters.iterations     = t : Total number of iterations 
%     Parameters.residualNorm   = norm(r) : Norm of the Residual at the final solution
%     Parameters.lambda         = Lambda  : Lambda value at the final solution
%     Parameters.activeColumnsCount = Number of Active Columns
%     Parameters.time           = Total Time spent
%
%Stat = TLARS Statistics Map for each iteration t
%     Stat(t).iteration     = Iteration t
%     Stat(t).residualNorm  = Norm of the residual at iteration t 
%     Stat(t).column        = Changed column at iteration t  
%     Stat(t).columnIndices = Factor indices of the added column
%     Stat(t).addColumn     = Add a column or remove a column at iteration t   
%     Stat(t).activeColumnsCount = length of the active columns at iteration t 
%     Stat(t).delta         = Delta at iteration t
%     Stat(t).lambda        = Lambda at iteration t
%     Stat(t).time          = Total elapsed time at iteration t


%% TLARS

tic
addpath(genpath('.\lib'));

%% Validating Input parameters
fprintf('Validating Input Attributes. \n');

algorithm = 'TLARS';

%Default Values
X = 0;
L0_Mode = false;
GPU_Computing = false;
Plot = false;                  %Plot norm of the residual at runtime
Debug_Mode = false;            %Save TLARS interal data
Path = '';                     %Path to save all variables in debug mode
Tolerence = 0.001;             %Default Tolerence
Iterations = numel(Y);         %Maximum Number of iteratons to run
Precision_factor = 5;          %5*eps - Round to 5 times the default machine precision(eps)

%Validate
validateattributes(D_Cell_Array,{'cell'},{'nonempty'},algorithm,'D_cell_Array',2);
validateattributes(Y,{'numeric'},{'nonempty','ndims', length(D_Cell_Array)},algorithm,'Y',1);
validateattributes(Active_Columns_Limit,{'numeric'},{'nonempty','positive'},algorithm,'Active_Columns_Limit',10);

tensor_dim_array = size(Y);
tensor_dim_array(tensor_dim_array <= 1) = [];
cellfun(@(Di, dl, idx) validateattributes(Di,{'numeric'},{'nonempty','nrows', dl},algorithm, strcat('Separable Dictionary','',num2str(idx)),2),D_Cell_Array, num2cell(tensor_dim_array),num2cell(1:max(length(tensor_dim_array))));

if nargin >= 4    
    Tolerence = varargin{1}; 
    validateattributes(Tolerence,{'numeric'},{'nonnegative','<=', 1},algorithm,'Tolerence',4);
end

if nargin >= 5
    X = varargin{2}; 
    validateattributes(X,{'numeric'},{'nonempty'},algorithm,'X',5);
end

if nargin >= 6 
    L0_Mode = varargin{3};
    validateattributes(L0_Mode,{'logical'},{'nonempty'},algorithm,'L0_Mode',6);
end

if nargin >= 7 
    GPU_Computing = varargin{4};
    validateattributes(GPU_Computing,{'logical'},{'nonempty'},algorithm,'GPU_Computing',7);

end
          
if nargin >= 8 
    Plot = varargin{5};
    validateattributes(Plot,{'logical'},{'nonempty'},algorithm,'Plot',8);
end

if nargin >= 9 
    Debug_Mode = varargin{6};
    validateattributes(Debug_Mode,{'logical'},{'nonempty'},algorithm,'Debug_Mode',9);
end

if nargin >= 10 
    Path = varargin{7};
    validateattributes(Path,{'char','string'},{},algorithm,'Path',10);
end

if nargin >= 11 
    Iterations = varargin{8};
    validateattributes(Iterations,{'numeric'},{'nonempty','positive'},algorithm,'Iterations',11);
end

if nargin >= 12 
    Precision_factor = varargin{9};
    validateattributes(Precision_factor,{'numeric'},{'nonempty','positive'},algorithm,'Precision_factor',12);
end

%% Define Variables
plot_frequency = 100; %After every 100 iterations plot norm_R and image
precision = Precision_factor*eps;

x = 0;
Active_Columns = [];

add_column_flag = -1;
changed_dict_column_index = -1;
changed_active_column_index = -1;
prev_t_added_dict_column_index = -1;
columnOperationStr = 'add';

order = length(D_Cell_Array);
core_tensor_dimensions = zeros(1,order);
gramian_cell_array = cell(1,order);
active_factor_column_indices = cell(1,order);

GInv = zeros(1000);   % Inverse of the Gramian

if nargout >= 5
    Stat = containers.Map('KeyType','int32','ValueType','any');
end

%% GPU computing requirments
if GPU_Computing
    if gpuDeviceCount > 0 
        fprintf('GPU Computing Enabled.\n\n');
        GInv = gpuArray(GInv);
    else
        fprintf(2,'GPU Device not found. GPU Computing Disabled.\n\n');
        GPU_Computing = false;
    end
end

%% Initialization
fprintf('Initializing %s... \n', algorithm);

%Normalize Y
tensor_norm = norm(vec(Y)); 
Y = Y./tensor_norm;

y = vec(Y); %vec(Y) returns the vectorization of Y
r = y;      %Initial residual r = y;
norm_R = norm(r);

%Normalize each column of every seperable dictionary D_n
for n=1:length(D_Cell_Array)    
    D_n=D_Cell_Array{n};
    D_Cell_Array(n) = {normc(D_n)};
    core_tensor_dimensions(n) = size(D_n,2);
end

%Calculate Separable Gram Matrices
fprintf('Calculating Separable Gram Matrices. \n');

for n = 1:order
    gramian_cell_array{n} = round(D_Cell_Array{n}'*D_Cell_Array{n},round(abs(log10(precision))));
end

%Check for a previous solution X
if nnz(X) > 0
    %If a previsous solution exists start from X    
     
    
    fprintf('Start TLARS calculations using the existing solution \n');
        
    %Calculate the residual tensor
    AX = fullMultilinearProduct( X, D_Cell_Array, false, GPU_Computing );
    R = Y - AX;
    r = vec(R);
    norm_R = norm(r);
    clear AX;    
   
     %Calculate the coeffiecint tensor and vectorize
    C = fullMultilinearProduct( R, D_Cell_Array, true, GPU_Computing ); % c = B'*r;
    c = gather(vec(C));
    clear R C;
    
    [lambda,changed_dict_column_index] = max(abs(c));
   
    %Find active columns
    x = vec(X); 
    Active_Columns = find(x ~= 0);
    x = x(Active_Columns); 
   
    fprintf('Number of Active_Columns = %d norm(r) = %d lambda = %d \n', length(Active_Columns),norm(r),lambda);
    
    fprintf('Obtaining the inverse of the Gramian \n');
    GI = getGramian( gramian_cell_array, Active_Columns, GPU_Computing );
    GInv = inv(GI);
    
    clear GI
    
    if Plot
        Ax = kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, x, false, GPU_Computing );
        show_tlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array );  
    end
    
else
    %If a previsous solution doesn't exists start from the begining
    %Calculate the initial correlation vector c
    fprintf('Calculating the Initial Correlation Vector. \n');

    C = fullMultilinearProduct( Y, D_Cell_Array, true, GPU_Computing ); % c = B'*r;
    c = gather(vec(C));
    clear C;

    [lambda,changed_dict_column_index] = max(abs(c)); %Initial lambda = max(c)

    % Set initial active column vector
    add_column_flag = 1;
    Active_Columns = changed_dict_column_index;
    changed_active_column_index = 1;
    prev_t_added_dict_column_index = changed_dict_column_index;   

end

%% TLARS Iterations

fprintf('Running %s Iterations... \n\n', algorithm);

for t=1:Iterations     
    try
    %% Calculate the inverse of the Gram matrix and update vector v
    
    % Obtain the sign sequence of the correlation of the active columns
    zI = sign(c(Active_Columns));

    %Calculate the inverse of the Gram Matrix for the active set. If A is the active columns matrix then Gramian GI = A'*A
    
    try
        [ dI, GInv ] = getDirectionVector(GInv, zI, gramian_cell_array, Active_Columns, add_column_flag, changed_dict_column_index, changed_active_column_index, core_tensor_dimensions, GPU_Computing );
        % If the direction vector has invalid entries, recalculate using
        %the recalculated invese of the Gramian.
        if any(isnan(dI))  
            fprintf('Gramian is singular. Using Inverse of the Gramian \n');
            GI = getGramian(gramian_cell_array, Active_Columns, GPU_Computing);
            GInv = inv(GI);
            dI = GInv*zI;
            clear GI;
        end  
    catch e        
        % In casee of an exception, disable GPU computing if enabled
        if GPU_Computing
            GPU_Computing = false;
            GInv = gather(GInv);
            zI = gather(zI);
            x = gather(x);
            fprintf(2,'Exception Occured. Disabling GPU Computing.\nException = %s \n', getReport(e));
            [ dI, GInv ] = getDirectionVector(GInv, zI, gramian_cell_array, Active_Columns, add_column_flag, changed_dict_column_index, changed_active_column_index, core_tensor_dimensions, GPU_Computing );
        else
            %If the GPU computing is disabled, recalculate the direction
            %vector using the recalculated inverse of the Gramian
            fprintf(2,'Exception Occured. Exception = %s \n', getReport(e));
            GI = getGramian(gramian_cell_array, Active_Columns, GPU_Computing);
            zI = gather(zI);
            GInv = inv(gather(GI));
            dI = GInv*zI;
            clear GI;
        end
    end                                   

    %Create vector v by selecting equivalent active columns from the Gramian G and multiplying with dI 
     v = kroneckerMatrixPartialVectorProduct( gramian_cell_array, Active_Columns, dI, false, GPU_Computing );% v= D'*A*dI

%% calculate delta_plus and delta_minus for every column 
    
    changed_dict_column_index = -1;
    delta  = -1;
    add_column_flag = 0;

    % Calculate delta_plus 
    delta_plus_1 = bsxfun(@minus, lambda, c)./bsxfun(@minus, 1, v);    
    delta_plus_1(Active_Columns) = inf;
    delta_plus_1(delta_plus_1 <= precision) = inf;
    [min_delta_plus_1, min_idx1] = min(delta_plus_1);

    delta_plus_2 = bsxfun(@plus, lambda, c)./bsxfun(@plus, 1, v);
    delta_plus_2(Active_Columns) = inf;
    delta_plus_2(delta_plus_2 <= precision) = inf;
    [min_delta_plus_2, min_idx2] = min(delta_plus_2);

    if min_delta_plus_1 < min_delta_plus_2
        changed_dict_column_index = gather(min_idx1);
        delta = full(min_delta_plus_1);
        add_column_flag = 1;
    else
        changed_dict_column_index = gather(min_idx2);
        delta = full(min_delta_plus_2);
        add_column_flag = 1;
    end      

    % Calculate delta_minus for L1 minimization
    if ~L0_Mode
        delta_minus  = -x./dI;
        delta_minus(delta_minus <= precision) = inf;
        [min_delta_minus, col_idx3] = min(delta_minus);
        min_idx3 = Active_Columns(col_idx3);

        if length(Active_Columns) > 1 && min_idx3 ~= prev_t_added_dict_column_index && min_delta_minus < delta
            changed_dict_column_index = gather(min_idx3);
            delta = full(min_delta_minus);
            add_column_flag = 0;
        end           
    end                                             

    %% Compute the solution x and parameters for next iteration

    % Check for invalid conditions
    if lambda < delta || lambda < 0 || delta < 0    
        fprintf('%s Stopped at: norm(r) = %d lambda = %d  delta = %d Time = %.3f\n',algorithm,nr,lambda,delta,toc);
        break;
    end

    %Update the solution x
    x = x + delta*dI;    
    lambda = lambda - delta; %lambda = max(c(active_columns));   
    c = c - delta*v; %c = B'*r;

    %Update the norm of the residual
    ad = gather(kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, dI, false, GPU_Computing));%v = A*dI; 
    r = r - delta*ad; %r = r - delta*A*dI;
    nr = norm(r);  
    norm_R = [norm_R; nr];
    
    % Calculate column indices of each changed column
    if Debug_Mode || nargout >= 5
        columnIndices = getKroneckerFactorColumnIndices( order, changed_dict_column_index, core_tensor_dimensions );
        active_factor_column_indices=cellfun(@(x,y) unique([x y]), active_factor_column_indices, columnIndices, 'UniformOutput', false);   
    end
    
    %Update the Stat class
    if nargout >= 5
        Stat(t) = StatClass( t,gather(nr),changed_dict_column_index,columnIndices,add_column_flag,length(Active_Columns),gather(delta),gather(lambda),toc );
    end    
     
    if Debug_Mode 
        fprintf('%s t = %d norm(r) = %d active columns = %d indices = %s %s column = %d Time = %.3f\n', algorithm, t, nr, length(Active_Columns), join(string(columnIndices)),columnOperationStr,changed_dict_column_index,toc);
    else
        fprintf('%s t = %d norm(r) = %d active columns = %d %s column = %d Time = %.3f\n',algorithm, t, nr, length(Active_Columns), columnOperationStr, changed_dict_column_index, toc);        
    end

    %Plot the current result
    if Plot && ( Debug_Mode || rem(t,plot_frequency) == 1 )
        Ax = kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, x, false, GPU_Computing );
        show_tlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array );  
    end   
    
    %Stopping criteria : If the stopping criteria is reached, stop the program and return results
    if nr < Tolerence || length(Active_Columns) >= Active_Columns_Limit
        fprintf('\n%s stopping criteria reached \n%s Finished at: norm(r) = %d lambda = %d  delta = %d tolerence = %d Time = %.3f\n',algorithm,algorithm,nr,lambda,delta,Tolerence,toc);
        break;
    end
    
    %% Add or remove column from the active set
    if  add_column_flag            
        Active_Columns = [Active_Columns; changed_dict_column_index];
        x = [x; 0];
        changed_active_column_index = length(x);
        prev_t_added_dict_column_index = changed_dict_column_index;
        columnOperationStr = 'add';
    else
        changed_active_column_index = find(Active_Columns == changed_dict_column_index);
        x(changed_active_column_index)  = [];
        Active_Columns(changed_active_column_index) = [];
        prev_t_added_dict_column_index = -1;
        columnOperationStr = 'remove';
    end    

    %Handle exception
    catch e
        if t >1
            
            X = constructCoreTensor(Active_Columns, x, core_tensor_dimensions);
            [x,X,Active_Columns,~,lambda,activeColumnsCount]=gather(x,X,Active_Columns,nr,lambda,length(Active_Columns));
            
            Ax = kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, x, false, GPU_Computing );
            
            if Debug_Mode && isdir(Path)
                
                [c,r,v,dI,zI,GInv,delta,Ax,norm_R]=gather(c,r,v,dI,zI,GInv,delta,Ax,norm_R);
                [delta_plus_1,delta_plus_2,delta_minus]=gather(delta_plus_1,delta_plus_2,delta_minus);
                [min_delta_plus_1,min_delta_plus_2,min_delta_minus]=gather(min_delta_plus_1,min_delta_plus_2,min_delta_minus);
                [min_idx1,min_idx2,min_idx3,col_idx3]=gather(min_idx1,min_idx2,min_idx3,col_idx3);            
                
                save(strcat(Path,algorithm,'_error_at_',num2str(t),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'_Results','.mat'),'Ax','Stat','Y','D_Cell_Array','Active_Columns', 'x', 'X','Parameters','-v7.3');
                save(strcat(Path,algorithm,'_error_at_',num2str(t),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'.mat'),'-v7.3');
            end
            if Plot
                show_tlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array ); 
            end
        end                 
        rethrow(e)
    end

end

%% Preparing TLARS Output

% construct the core tensor
X = constructCoreTensor(Active_Columns, x, core_tensor_dimensions);
[x,X,Active_Columns,nr,lambda,activeColumnsCount]=gather(x,X,Active_Columns,nr,lambda,length(Active_Columns));

% Set output variables based on the number of requested outputs
if nargout >=1
    varargout{1} = X;
end
if nargout >=2
    varargout{2} = Active_Columns;
end
if nargout >=3
    varargout{3} = x;
end
if nargout >=4    
    varargout{4} = Parameters(t,nr,lambda,activeColumnsCount,toc);
end
if nargout >=5
    varargout{5} = Stat;
end

if Plot || Debug_Mode
    
    Ax = kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, x, false, GPU_Computing );        
    
    if Plot
        show_tlars( Ax, Y, tensor_norm, norm_R, tensor_dim_array );
    end
end

if Debug_Mode && isdir(Path)
    
    [c,r,v,dI,zI,GInv,delta,Ax,norm_R]=gather(c,r,v,dI,zI,GInv,delta,Ax,norm_R);
    [delta_plus_1,delta_plus_2,delta_minus]=gather(delta_plus_1,delta_plus_2,delta_minus);
    [min_delta_plus_1,min_delta_plus_2,min_delta_minus]=gather(min_delta_plus_1,min_delta_plus_2,min_delta_minus);
    [min_idx1,min_idx2,min_idx3,col_idx3]=gather(min_idx1,min_idx2,min_idx3,col_idx3);
    
    save(strcat(Path, algorithm,'_Finished at_',num2str(t),'.mat'),'-v7.3');
end
    
end

