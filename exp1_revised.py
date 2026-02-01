"""
@author: Sangeeta Yadav
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import scipy.io
import sys
import time
import pyvista as pv

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

        [self.t_data, self.x_data, self.y_data, self.z_data, self.c_data] = \
            [t_data, x_data, y_data, z_data, c_data]

        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = \
            [t_eqns, x_eqns, y_eqns, z_eqns]

        # Placeholders
        [self.t_data_tf, self.x_data_tf, self.y_data_tf,
         self.z_data_tf, self.c_data_tf] = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(5)]

        [self.t_eqns_tf, self.x_eqns_tf,
         self.y_eqns_tf, self.z_eqns_tf] = \
            [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]

        # Neural Network
        self.net_cuvwp = neural_net(self.t_data,
                                   self.x_data,
                                   self.y_data,
                                   self.z_data,
                                   layers=self.layers)

        [self.c_data_pred,
         self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)

        [self.c_eqns_pred,
         self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
                                            self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)

        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred,
         self.e5_eqns_pred] = Navier_Stokes_3D(
             self.c_eqns_pred,
             self.u_eqns_pred,
             self.v_eqns_pred,
             self.w_eqns_pred,
             self.p_eqns_pred,
             self.t_eqns_tf,
             self.x_eqns_tf,
             self.y_eqns_tf,
             self.z_eqns_tf,
             self.Pec,
             self.Rey
         )

        # Loss
        self.loss = mean_squared_error(self.c_data_pred, self.c_data_tf) + \
                    mean_squared_error(self.e1_eqns_pred, 0.0) + \
                    mean_squared_error(self.e2_eqns_pred, 0.0) + \
                    mean_squared_error(self.e3_eqns_pred, 0.0) + \
                    mean_squared_error(self.e4_eqns_pred, 0.0) + \
                    mean_squared_error(self.e5_eqns_pred, 0.0)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = tf_session()

    # ------------------------------------------------------------------

    def train(self, max_epochs, learning_rate, error_epochs,
              t_test, x_test, y_test, z_test,
              c_test, u_test, v_test, w_test, p_test):

        N_data = self.t_data.shape[0]
        N_eqns = self.t_eqns.shape[0]

        error_history = {
            'epoch': [],
            'c': [],
            'u': [],
            'v': [],
            'w': [],
            'p': []
        }

        print('Epoch, Loss')

        for it in range(1, max_epochs + 1):

            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)

            tf_dict = {
                self.t_data_tf: self.t_data[idx_data, :],
                self.x_data_tf: self.x_data[idx_data, :],
                self.y_data_tf: self.y_data[idx_data, :],
                self.z_data_tf: self.z_data[idx_data, :],
                self.c_data_tf: self.c_data[idx_data, :],
                self.t_eqns_tf: self.t_eqns[idx_eqns, :],
                self.x_eqns_tf: self.x_eqns[idx_eqns, :],
                self.y_eqns_tf: self.y_eqns[idx_eqns, :],
                self.z_eqns_tf: self.z_eqns[idx_eqns, :],
                self.learning_rate: learning_rate
            }

            self.sess.run(self.train_op, tf_dict)

            if it % 100 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('%d, %.3e' % (it, loss_value))
                sys.stdout.flush()

            if it in error_epochs:
                c_pred, u_pred, v_pred, w_pred, p_pred = \
                    self.predict(t_test, x_test, y_test, z_test)

                error_history['epoch'].append(it)
                error_history['c'].append(relative_error(c_pred, c_test))
                error_history['u'].append(relative_error(u_pred, u_test))
                error_history['v'].append(relative_error(v_pred, v_test))
                error_history['w'].append(relative_error(w_pred, w_test))
                error_history['p'].append(
                    relative_error(p_pred - np.mean(p_pred),
                                   p_test - np.mean(p_test))
                )

                print(f"Errors saved at epoch {it}")

        return error_history

    # ------------------------------------------------------------------

    def predict(self, t_star, x_star, y_star, z_star):

        tf_dict = {
            self.t_data_tf: t_star,
            self.x_data_tf: x_star,
            self.y_data_tf: y_star,
            self.z_data_tf: z_star
        }

        c_star = self.sess.run(self.c_data_pred, tf_dict)
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)

        return c_star, u_star, v_star, w_star, p_star


# ======================================================================
# =============================== MAIN =================================
# ======================================================================

if __name__ == "__main__":

    batch_size = 10000
    layers = [4] + 10 * [5 * 50] + [5]

    data = scipy.io.loadmat('../Data/sortedfifty_data.mat')

    t_star = data['t_star']
    x_star = data['x_star']
    y_star = data['y_star']
    z_star = data['z_star']

    U_star = data['U_star']
    V_star = data['V_star']
    W_star = data['W_star']
    P_star = data['P_star']
    C_star = data['C_star']

    T = t_star.shape[0]
    N = x_star.shape[0]

    T_star = np.tile(t_star, (1, N)).T
    X_star = np.tile(x_star, (1, T))
    Y_star = np.tile(y_star, (1, T))
    Z_star = np.tile(z_star, (1, T))

    idx_t = np.arange(T)
    idx_x = np.arange(N)

    t_data = T_star[:, idx_t][idx_x, :].flatten()[:, None]
    x_data = X_star[:, idx_t][idx_x, :].flatten()[:, None]
    y_data = Y_star[:, idx_t][idx_x, :].flatten()[:, None]
    z_data = Z_star[:, idx_t][idx_x, :].flatten()[:, None]
    c_data = C_star[:, idx_t][idx_x, :].flatten()[:, None]

    t_eqns = t_data.copy()
    x_eqns = x_data.copy()
    y_eqns = y_data.copy()
    z_eqns = z_data.copy()

    model = HFM(t_data, x_data, y_data, z_data, c_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                layers, batch_size,
                Pec=1.0 / 0.0101822,
                Rey=1.0 / 0.0101822)

    snap = np.array([10])
    t_test = T_star[:, snap]
    x_test = X_star[:, snap]
    y_test = Y_star[:, snap]
    z_test = Z_star[:, snap]

    c_test = C_star[:, snap]
    u_test = U_star[:, snap]
    v_test = V_star[:, snap]
    w_test = W_star[:, snap]
    p_test = P_star[:, snap]

    error_epochs = [2000, 4000, 6000, 8000, 10000]

    error_history = model.train(
        max_epochs=10000,
        learning_rate=1e-3,
        error_epochs=error_epochs,
        t_test=t_test,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        c_test=c_test,
        u_test=u_test,
        v_test=v_test,
        w_test=w_test,
        p_test=p_test
    )

    scipy.io.savemat(
        "../Results/Aneurysm3D/error_vs_epoch.mat",
        {
            'epoch': np.array(error_history['epoch']),
            'error_c': np.array(error_history['c']),
            'error_u': np.array(error_history['u']),
            'error_v': np.array(error_history['v']),
            'error_w': np.array(error_history['w']),
            'error_p': np.array(error_history['p'])
        }
    )