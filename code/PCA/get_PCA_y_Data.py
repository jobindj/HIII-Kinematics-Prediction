# -*- coding: utf-8 -*-
"""

@author: gurram
"""
def get_matrix(y, idx, Nd, ts):
    """

    Parameters
    ----------
    samples_xarr_disk : Xarray
        4D representation of the simulation data
        Shape: (n_simulations, n_nodes, n_DoFs, n_states)
    X : Array
        2D array of the FE parameters of simulations
        Shape: (n_simulations, n_FE_parameters)
    train_idx : list
        List of training set indices of simulations
    test_idx : list
        List of test set indices of simulations
    Nd : list
        List of node IDs of the HBM/dummy
    ts : int
        Integer indicating the number of sates/timesteps of the 
        results of each simulation

    Returns
    -------
    Matrix1_XRlass: DataFrame
        Simulation results/displacements arraged as per matrix type 1
        Shape: (n_nodes*n_train_simulations, n_states*n_DoFs)
    Matrix2_XRlass: DataFrame
        Simulation results/displacements arraged as per matrix type 2
        Shape: (n_states*n_train_simulations, n_nodes*n_DoFs)

    """
    import numpy as np
    import pandas as pd    

## Build data matrices by rearranging displacement data from all simulations considered
    
    x_slice = y.sel(DoFs = 'x_disp')
    y_slice = y.sel(DoFs = 'y_disp')
    z_slice = y.sel(DoFs = 'z_disp')
    
    x_transp = x_slice.transpose("nodes",...)
    y_transp = y_slice.transpose("nodes",...)
    z_transp = z_slice.transpose("nodes",...)
    
    x_disp_stack= x_transp.stack(z=("sims","nodes")).values
    y_disp_stack= y_transp.stack(z=("sims","nodes")).values
    z_disp_stack= z_transp.stack(z=("sims","nodes")).values
    
    Matrix_XRlass = np.concatenate((x_disp_stack, y_disp_stack, z_disp_stack), axis=0) 
    
    Matrix1_XRlass = Matrix_XRlass.T
    
    x_disp_stack2 = np.reshape((x_slice.values),(((len(idx)*ts),(len(Nd)))))
    y_disp_stack2 = np.reshape((y_slice.values),(((len(idx)*ts),(len(Nd)))))
    z_disp_stack2 = np.reshape((z_slice.values),(((len(idx)*ts),(len(Nd)))))
    
    Matrix2_XRlass= np.concatenate((x_disp_stack2, y_disp_stack2, z_disp_stack2), axis=1)
    
    
    return Matrix1_XRlass, Matrix2_XRlass;

def get_PCA_y_Data(train_idx, Matrix, stan, n_PCs, ts):
    """

    Parameters
    ----------
    train_idx : list
        List of indices of the simulations from xarray of simulation data
    Matrix : DataFram
        2D matrix of displacement data from training set of simulations
        Type 1: (n_nodes*n_train_simulations, n_states*n_DoFs)
        Type 2: (n_states*n_train_simulations, n_nodes*n_DoFs)
    stan : True or False
        To standardize the data or not w.r.t mean and standard deviation of the data
    n_PCs : int or float
        The number of principal components to consider for PCA
        int to specify number of PCs 
        float to specify percentage of variance to be accounted within the PCs
    ts : int
        Number of timestates of data recorded in D3plots

    Returns
    -------
    y_PCA_ReSh : DataFrame
        Data from PCA with 'n_comps' of principal components arranged in 
        shape: (n_train_simulations, n_principal_components) 
    n_comps : int
        Number of PCs considered for PCA
    y_PCA_InvTrans_Data : Array of Float32
        The inverse transformed data from PCA
    cumulative_variance : Array of Float32
        The variance values of individual Principal components

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

## Standardization (If needed)
    scaler_std = StandardScaler(with_mean=stan, with_std=stan) #with_mean=True, with_std=True
    scaler_std = scaler_std.fit(Matrix)
    y_Data_std = scaler_std.transform(Matrix)
    
## Determine variance captured by the PCs
    varience = PCA().fit(y_Data_std).explained_variance_

## Calculation of cumulative variance to determine number of principle components needed
    cumulative_variance = np.cumsum(PCA().fit(y_Data_std).explained_variance_ratio_)

## Perform PCA and calulate error from reconstructed values based on no. of PCs        
    pca = PCA(n_components = n_PCs)
    y_PCA = pca.fit_transform(pd.DataFrame(y_Data_std))
    n_comps = pca.n_components_
    y_PCA_inverse = pca.inverse_transform(y_PCA) # Inverse transfrom of sample data as per PCs to original form
    y_PCA_InvTrans_Data = scaler_std.inverse_transform(y_PCA_inverse, copy=None) # inverse standerdization
    pca_mse = mean_squared_error(Matrix,y_PCA_InvTrans_Data)
    pca_mae = mean_absolute_error(Matrix,y_PCA_InvTrans_Data)
    print("The weights of {} PCs: ".format(varience[0:n_comps]))
    print("MSE: {} for n_comps: {}".format(pca_mse, n_comps))
    print("MAE: {} for n_comps: {}".format(pca_mae, n_comps))

    
## Flattening PCA scores according to n_samples. For training
    reshape_y_pca2 =[]
    pd.DataFrame(reshape_y_pca2)
    for i in range(0,len(train_idx)):
#         temp_1 = y_PCA[(i*31+i):(i*31+i+32)].T
        temp_1 = y_PCA[(i*(ts-1)+i):(i*(ts-1)+i+ts)].T
        temp_2 = temp_1.flatten()
        temp = pd.DataFrame(temp_2)
        reshape_y_pca2.append(temp.T)

    y_PCA_ReSh = pd.DataFrame(np.row_stack(reshape_y_pca2), index = train_idx)       
    
    return y_PCA_ReSh, n_comps, scaler_std, pca, y_PCA_InvTrans_Data, cumulative_variance;





