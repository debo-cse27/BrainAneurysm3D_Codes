"""
Experiment 2: Learning Rate Variation
Variations: 1e-2, 5e-3, 1e-3, 5e-4, 1e-4
Fixed Training Time: 2 hours
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import sys
import os
import pyvista as pv
from pyevtk.hl import gridToVTK

from utilities import neural_net, Navier_Stokes_3D, \
                    tf_session, mean_squared_error, relative_error

class HFM(object):
    def __init__(self, t_data, x_data, y_data, z_data, c_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       layers, batch_size,
                       Pec, Rey):
        self.layers = layers
        self.batch_size = batch_size
        self.Pec = Pec
        self.Rey = Rey
        
        [self.t_data, self.x_data, self.y_data, self.z_data, self.c_data] = [t_data, x_data, y_data, z_data, c_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
        
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf, self.c_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        
        self.net_cuvwp = neural_net(self.t_data, self.x_data, self.y_data, self.z_data, layers = self.layers)
        
        [self.c_data_pred, self.u_data_pred, self.v_data_pred, self.w_data_pred, self.p_data_pred] = \
            self.net_cuvwp(self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf)
         
        [self.c_eqns_pred, self.u_eqns_pred, self.v_eqns_pred, self.w_eqns_pred, self.p_eqns_pred] = \
            self.net_cuvwp(self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf)
        
        [self.e1_eqns_pred, self.e2_eqns_pred, self.e3_eqns_pred, self.e4_eqns_pred, self.e5_eqns_pred] = \
            Navier_Stokes_3D(self.c_eqns_pred, self.u_eqns_pred, self.v_eqns_pred, self.w_eqns_pred, self.p_eqns_pred,
                             self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf,
                             self.Pec, self.Rey)
        
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0)
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.sess = tf_session()
    
    def train(self, total_time, learning_rate):
        
        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        print('It, Loss, Time(s), Running Time(h), Learning Rate')
        while running_time < total_time:
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch, x_data_batch, y_data_batch, z_data_batch, c_data_batch) = \
                (self.t_data[idx_data,:], self.x_data[idx_data,:], self.y_data[idx_data,:], self.z_data[idx_data,:], self.c_data[idx_data,:])

            (t_eqns_batch, x_eqns_batch, y_eqns_batch, z_eqns_batch) = \
                (self.t_eqns[idx_eqns,:], self.x_eqns[idx_eqns,:], self.y_eqns[idx_eqns,:], self.z_eqns[idx_eqns,:])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.c_data_tf: c_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                [loss_value] = self.sess.run([self.loss], tf_dict)
                print('%d,%.3e, %.2f, %.2f, %.1e' %(it, loss_value, elapsed, running_time, learning_rate))
                sys.stdout.flush()
                start_time = time.time()
            it += 1
    
    def predict(self, t_star, x_star, y_star, z_star):
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        return c_star, u_star, v_star, w_star, p_star

if __name__ == "__main__": 
    results_dir = "../Results/Experiment2_LR"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    
    lr_variations = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    
    layers = [4] + 10*[5*50] + [5]
    
    data = scipy.io.loadmat('../Data/sortedfifty_data.mat')
        
    t_star = data['t_star']; x_star = data['x_star']; y_star = data['y_star']; z_star = data['z_star']
    T = t_star.shape[0]; N = x_star.shape[0]
        
    U_star = data['U_star']; V_star = data['V_star']; W_star = data['W_star']
    P_star = data['P_star']; C_star = data['C_star']    
    
    T_star = np.tile(t_star, (1,N)).T
    X_star = np.tile(x_star, (1,T))
    Y_star = np.tile(y_star, (1,T))
    Z_star = np.tile(z_star, (1,T))
    
    T_data = T
    N_data = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_data, replace=False)
    
    t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_data = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    T_eqns = T
    N_eqns = N
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t][idx_x,:].flatten()[:,None]
    
    for lr in lr_variations:
        print(f"\n======== STARTING RUN: LR = {lr} ========\n")
        tf.reset_default_graph()
        
        model = HFM(t_data, x_data, y_data, z_data, c_data,
                    t_eqns, x_eqns, y_eqns, z_eqns,
                    layers, batch_size=10000,
                    Pec = 1.0/0.0101822, Rey = 1.0/0.0101822)
        
        model.train(total_time=2.0, learning_rate=lr)
        
        snap = 10
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        c_test = C_star[:,snap:snap+1]
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
        
        c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))

        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        print('Error p: %e' % (error_p))
            
        C_pred_all = 0*C_star
        U_pred_all = 0*U_star
        V_pred_all = 0*V_star
        W_pred_all = 0*W_star
        P_pred_all = 0*P_star
        
        vtk_dir = f"{results_dir}/vtk_lr_{lr}"
        if not os.path.exists(vtk_dir): os.makedirs(vtk_dir)

        for snap_idx in range(0, t_star.shape[0]):
            t_test = T_star[:,snap_idx:snap_idx+1]
            x_test = X_star[:,snap_idx:snap_idx+1]
            y_test = Y_star[:,snap_idx:snap_idx+1]
            z_test = Z_star[:,snap_idx:snap_idx+1]
            
            c_test = C_star[:,snap_idx:snap_idx+1]
            u_test = U_star[:,snap_idx:snap_idx+1]
            v_test = V_star[:,snap_idx:snap_idx+1]
            w_test = W_star[:,snap_idx:snap_idx+1]
            p_test = P_star[:,snap_idx:snap_idx+1]
        
            c_pred, u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
            
            C_pred_all[:,snap_idx:snap_idx+1] = c_pred
            U_pred_all[:,snap_idx:snap_idx+1] = u_pred
            V_pred_all[:,snap_idx:snap_idx+1] = v_pred
            W_pred_all[:,snap_idx:snap_idx+1] = w_pred
            P_pred_all[:,snap_idx:snap_idx+1] = p_pred
        
            points = np.hstack((x_test, y_test, z_test))
            grid = pv.PolyData(points)

            grid.point_data["C_test"] = c_test
            grid.point_data["U_test"] = u_test
            grid.point_data["V_test"] = v_test
            grid.point_data["W_test"] = w_test
            grid.point_data["P_test"] = p_test
            
            grid.point_data["C_pred"] = c_pred
            grid.point_data["U_pred"] = u_pred
            grid.point_data["V_pred"] = v_pred
            grid.point_data["W_pred"] = w_pred
            grid.point_data["P_pred"] = p_pred
            
            grid.point_data["Error_C"] = c_pred - c_test
            grid.point_data["Error_U"] = u_pred - u_test
            grid.point_data["Error_V"] = v_pred - v_test
            grid.point_data["Error_W"] = w_pred - w_test
            grid.point_data["Error_P"] = p_pred - p_test

            grid.save(f"{vtk_dir}/allresult_{snap_idx}.vtk")
        
        scipy.io.savemat(f"{results_dir}/results_lr_{lr}.mat",
                         {'C_pred':C_pred_all, 'U_pred':U_pred_all, 'V_pred':V_pred_all, 'W_pred':W_pred_all, 'P_pred':P_pred_all})