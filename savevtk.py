"""
@author: Sangeeta Yadav
"""
import numpy as np
import pyvista as pv

import scipy.io
import time
import sys


if __name__ == "__main__": 
    
    # Load Data
    data = scipy.io.loadmat('../Data/sortedfifty_data.mat')
        
    t_star = data['t_star'] # T x 1
    x_star = data['x_star'] # N x 1
    y_star = data['y_star'] # N x 1
    z_star = data['z_star'] # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]
        
    U_star = data['U_star'] # N x T
    V_star = data['V_star'] # N x T
    W_star = data['W_star'] # N x T
    P_star = data['P_star'] # N x T
    C_star = data['C_star'] # N x T    
    
    # Rearrange Data 
    T_star = np.tile(t_star, (1,N)).T # N x T
    X_star = np.tile(x_star, (1,T)) # N x T
    Y_star = np.tile(y_star, (1,T)) # N x T
    Z_star = np.tile(z_star, (1,T)) # N x T
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    
    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    
    # Test Data
    snap = np.array([15])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    z_test = Z_star[:,snap]
    
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    w_test = W_star[:,snap]
    p_test = P_star[:,snap]
    
    
    
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
    
        grid = pv.StructuredGrid(x_test, y_test, z_test)

        # Add the pressure data as a point array
        grid.point_data["Pressure"] = p_test
        grid.point_data["Concentration"] = c_test
        grid.point_data["U"] = u_test
        grid.point_data["V"] = v_test
        grid.point_data["W"] = w_test
        grid.point_data["T"] = t_test

        # Save the grid to a VTK file
        grid.save("../Results/Aneurysm3D_GroundTruth/checkvtktwo/allresult_%s.vtk" %snap)
    
        
       