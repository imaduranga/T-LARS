%Run TLARS v1.0.0
%Author : Ishan Wickramsingha
%Date : 2019/10/18

%% References

% If you use this code in a scientific publication, please cite following papers:

% Wickramasingha, Ishan, Ahmed Elrewainy, Michael Sobhy, and Sherif S. Sherif. " Tensor Least Angle Regression for Sparse Representations of Multi-dimensional Signals." Neural computation (2020) 
% Elrewainy, A., & Sherif, S. S. (2019). Kronecker least angle regression for unsupervised unmixing of hyperspectral imaging data. Signal, Image and Video Processing, 14(2), 359–367. https://doi.org/10.1007/s11760-019-01562-w


%%

clear;clc;

data = 'data_2D_mri';
save_data = false;

Y = 0;                              %Data
D_Cell_Array = {};                  %Dictionary matrices(factor matrices) as a cell array
Active_Columns_Limit = 8000;        %Depends on the required sparsity
Tolerence = 0.01;                   %0.01 tolerence (If Active_Columns_Limit not reached, stop T-LARS when norm of the residual error reach tolerence)
X = 0;                              %Previous Solution
L0_Mode = false;                    %True for L0 or false for L1 Minimization
GPU_Computing = true;               %If True run on GPU if available
Plot = true;                        %Plot norm of the residual at runtime
Debug_Mode = false;                 %Save TLARS variable into a .mat file given in path in debug mode 
Path = '.\example\results\';        %Path to save all variables in debug mode

Iterations = 1e6;                   %Maximum Number of iteratons to run
Precision_factor = 5;               %Round to e^-20

algorithm = 'TLARS';
%%
LP = 'L1';
if L0_Mode
    LP = 'L0';
end

if save_data || Debug_Mode
    Path = strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'_',datestr(now,'yyyymmdd_HHMM'),'\');
    mkdir(Path);
    diary(strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'.log'));
    diary on
    %profile on
end

fprintf('Running %s for %s until norm of the residual reach %d%%  \n\n',algorithm, data, Tolerence);
str = '';
load(strcat('.\example\data\',data,'.mat'));
fprintf('Dictionary = %s \n\n',str);

[ X, Active_Columns, x, Parameters, Stat ] = TLARS( Y, D_Cell_Array, Active_Columns_Limit, Tolerence, X, L0_Mode, GPU_Computing, Plot, Debug_Mode, Path, Iterations, Precision_factor );

%% Test

for n=1:length(D_Cell_Array)
    D_n=D_Cell_Array{n};
    D_Cell_Array(n) = {normc(D_n)};
end

Ax = kroneckerMatrixPartialVectorProduct( D_Cell_Array, Active_Columns, x, false, GPU_Computing );
Ax = gather(Ax);

y = normc(vec(Y));
r = y - Ax;

fprintf('\nTLARS Completed. \nNorm of the Residual = %g \n', norm(r));

%%
if save_data || Debug_Mode
    diary off
    %profile off

    save(strcat(Path,algorithm,'_',LP,'_',data,'_Results','.mat'),'Ax','Parameters','Stat','Y','D_Cell_Array','Active_Columns', 'x', 'X','-v7.3');
    save(strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS'),'_GPU.mat'),'-v7.3');    
    
    f = gcf;
    savefig(f,strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'.fig'));
    saveas(f,strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'.jpg'));
    %profsave(profile('info'),strcat(Path,algorithm,'_',LP,'_',data,'_',num2str(Tolerence),'_',datestr(now,'dd-mmm-yyyy_HH-MM-SS')));
end



