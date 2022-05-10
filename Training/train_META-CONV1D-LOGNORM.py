import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import models
import tensorflow.keras as keras
import math
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(0)
random.seed(0)
n_workers = 10


IDS_TRAIN = []
IDS_TRAIN.append([12, 0, 1, 11])
IDS_TRAIN.append([7, 0, 17, 11])
IDS_TRAIN.append([12, 5, 15, 9])
IDS_TRAIN.append([8, 22, 3, 11])
IDS_TRAIN.append([7, 0, 14, 9])

if not len(IDS_TRAIN):
    n_combs = 5
else:
    n_combs = len(IDS_TRAIN)

params = {}
params["SCRIPT"] = "train_META-CONV1D-LOGNORM.py"
params["path_data"] = []
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/")
params["path_data"].append("../Dataset_Collection/Datasets/Exp_2021-08-04 16-58-16/")


params["FREQ_VAL"] = 2
params["MAX_NO_IMPROVEMENT_EPOCHS"] = 12
params["EPOCHS"] = 60
params["BATCH_SIZE"] = 32
params["LOSS"] = 'mse'
params["LEARNING_RATE"] = 1e-4
params["N_ITERATIONS_TRAIN"] = 0.25
params["N_ITERATIONS_VAL"] = 0.0625
params["MODEL"] = "model_META-CONV1D-LOGNORM"
params["N_COMPONENTS"] = 0
params["ESTIMATION_TYPE"] = "probability" #probability or deterministic
params["METRICS"] = True

params["REMOVE_ROUGH"] = False

params["EXTRA_INFO_NORM_TYPE"] = ""
params["ENERGY_NORM_TYPE"] = "standardize_shift"
params["EPS"] = 1e-3

params["LENGTH_SEQUENCE"] = 3
params["MERGED_MODEL"] = True
params["LENGTH_PAST"] = 0
params["LENGTH_SHOTS"] = 3
params["LENGTH_META"] = 3
params["N_SHOTS"] = 3
params["N_META"] = 1
params["MERGED_SHOTS_GEOM"] = True
params["MERGED_META_GEOM"] = True
params["MERGED_SHOTS_OUTPUT"] = True
params["MERGED_META_OUTPUT"] = True

params["path_sum_indices"] = []

params["path_sum_indices"] = []
params["path_sum_indices"].append("../Dataset_Collection/Datasets/Exp_2021-08-01 15-18-24/Merged_Data/sum_indices.csv")
params["path_sum_indices"].append("../Dataset_Collection/Datasets/EExp_2021-08-04 16-58-16/Merged_Data/sum_indices.csv")


params["TRAIN_PERC"] = 1
params["LOG_DIR"] = "./Exp00/log_meta_conv1d_lognorm/"

params["BATCH_TYPE"] = "mixed"
params["INPUT_FEATURES"] = ["wheel_trace"]
params["WHEEL_TRACE_SHAPE"] = (78, 40)
params["REMOVE_CENTRAL_BAND"] = 0

# params["SHOTS_EXTRA_INFO"] = ["mean_pitch_meas", "mean_roll_meas", "var_pitch_meas", "var_roll_meas",
#                               "initial_speed_long", "mean_speed_long", "max_speed_long", "min_speed_long",
#                               "initial_speed_lat", "mean_speed_lat", "max_speed_lat", "min_speed_lat"]

params["SHOTS_EXTRA_INFO"] = []


CATEGORIES = []
CATEGORIES.append([7, 8, 10, 12])  # Clay high moisture content
CATEGORIES.append([5, 0, 22])  # Loose frictional
CATEGORIES.append([1, 3, 4, 13, 14, 15, 16, 17])  # Compact frictional
CATEGORIES.append([9, 11])  # Dry clay


params["TRAINING_DATASETS_PER_CATEGORY"] = [1, 1, 1, 1]

n_metrics = params["N_SHOTS"]-1 + (params["N_SHOTS"]-1)*(params["LENGTH_SHOTS"]-1)*(int(not params["MERGED_SHOTS_OUTPUT"])) + \
    params["N_META"] + params["N_META"] * \
    (params["LENGTH_META"]-1)*(int(not params["MERGED_META_OUTPUT"]))
params["SHOTS_WEIGHTS"] = [1]*n_metrics


def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    return tf.subtract(np.float64(1.0), tf.divide(residual, total))

def MSE(y, y_pred):
    return tf.reduce_mean((tf.square(tf.subtract(y, y_pred))))

class DataSequence(keras.utils.Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    modified from: https://stackoverflow.com/questions/51843149/loading-batches-of-images-in-keras-from-pandas-dataframe
    """

    def __init__(self, df, df_idx, batch_size,  mode='train'):
        self.len_df = len(df)
        self.batch_size = batch_size
        self.mode = mode  # shuffle when in train mode
        self.df = {}
        self.df_idx = {}
        self.terrain_ids = list(df["terrain_id"].drop_duplicates().values)
        self.datasets = list(df["dataset"].drop_duplicates().values)
        for terrain_id in self.terrain_ids:
            self.df["{}".format(terrain_id)] = {}
            dfi = df[df.terrain_id==terrain_id]
            self.df_idx["{}".format(terrain_id)] = df_idx[df_idx.terrain_id==terrain_id]
            for dataset in self.datasets:
                self.df["{}".format(terrain_id)]["{}".format(dataset)] = dfi[dfi.dataset==dataset]

        self.on_epoch_end()

    def __len__(self):
        # compute number of batches to yield
        if self.mode == 'train':
            return int(math.ceil(self.len_df / float(self.batch_size))*params["N_ITERATIONS_TRAIN"])
        elif self.mode == 'validation':
            return int(math.ceil(self.len_df / float(self.batch_size))*params["N_ITERATIONS_VAL"])

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(self.len_df)
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def __getitem__(self, index):
        for i in range(self.batch_size):
            terrain_id = random.sample(self.terrain_ids,1)[0]
            dfx = self.df_idx["{}".format(terrain_id)]
            
            if params["BATCH_TYPE"] == "mixed" and len(self.datasets)>1:
                if random.random() > 1/3:
                    dataset_id = random.sample(self.datasets, 1)[0]
                    dfx = dfx[dfx["dataset"]==dataset_id]

            row = dfx.sample(n=params["N_SHOTS"]+params["N_META"])
            row_shots = row.sample(n=params["N_SHOTS"])
            row_meta = row.drop(row_shots.index)
            
            if len(self.datasets)>1:
                shot_tot = pd.DataFrame()
                for (k,d) in zip(row_shots.k.values,row_shots.dataset.values):
                    shot = self.df["{}".format(terrain_id)]["{}".format(d)].iloc[[k+z for z in range(params["LENGTH_SHOTS"])]]
                    shot_tot = pd.concat([shot_tot,shot])
                meta_tot = pd.DataFrame()
                for (k,d) in zip(row_meta.k.values,row_meta.dataset.values):
                    meta = self.df["{}".format(terrain_id)]["{}".format(d)].iloc[[k+z for z in range(params["LENGTH_META"])]]
                    meta_tot = pd.concat([meta_tot,meta])
            else:
                idx = []
                for k in row_shots.k.values:
                    idx.extend(k+z for z in range(params["LENGTH_SHOTS"]))
                shot_tot = self.df["{}".format(terrain_id)]["0"].iloc[idx]
                idx = []
                for k in row_meta.k.values:
                    idx.extend(k+z for z in range(params["LENGTH_META"]))
                meta_tot = self.df["{}".format(terrain_id)]["0"].iloc[idx]
            

            xe_shots = shot_tot.loc[:, ["energy"]].values
            ye_meta = meta_tot.loc[:, ["energy"]].values
            
            if params["SHOTS_EXTRA_INFO"]:
                xei_shots = shot_tot.loc[:, params["SHOTS_EXTRA_INFO"]].values

            xg_shots_string = shot_tot.loc[:, ["wheel_trace"]].values
            for p in range(len(xg_shots_string)):
                xg = np.array([float(v) for v in xg_shots_string[p][0].split(' ')]).reshape(params["WHEEL_TRACE_SHAPE"])
                if not p:
                    xg_shots = np.expand_dims(xg, axis=0)
                else:
                    xg_shots = np.concatenate(
                        [xg_shots, np.expand_dims(xg, axis=0)], axis=0)
            xg_meta_string = meta_tot.loc[:, ["wheel_trace"]].values
            for p in range(len(xg_meta_string)):
                xg = np.array([float(v) for v in xg_meta_string[p][0].split(' ')]).reshape(params["WHEEL_TRACE_SHAPE"])
                if not p:
                    xg_meta = np.expand_dims(xg, axis=0)
                else:
                    xg_meta = np.concatenate(
                        [xg_meta, np.expand_dims(xg, axis=0)], axis=0)

            if params["MERGED_SHOTS_GEOM"]:
                xg_shots_tot = np.empty((params["N_SHOTS"],
                                         xg_shots.shape[-2]+16 *
                                         (params["LENGTH_SHOTS"]-1),
                                         xg_shots.shape[-1]))
                zrel = shot_tot.loc[:, ["zrel"]].values
                for z in range(params["N_SHOTS"]):
                    # Merge the shots in a single terrain trace
                    xg_shots_tot[z, :xg_shots.shape[-2]
                                 ] = xg_shots[z*params["LENGTH_SHOTS"]]
                    for p in range(1, params["LENGTH_SHOTS"]):
                        xg_shots_tot[z, xg_shots.shape[-2]+16*(p-1):xg_shots.shape[-2]+16*p] = xg_shots[p+z *
                                                                                                        params["LENGTH_SHOTS"], -16:]+zrel[p+z*params["LENGTH_SHOTS"]]-zrel[z*params["LENGTH_SHOTS"]]
            else:
                xg_shots_tot = xg_shots
            if params["MERGED_META_GEOM"]:
                xg_meta_tot = np.empty((params["N_META"],
                                        xg_meta.shape[-2]+16 *
                                        (params["LENGTH_META"]-1),
                                        xg_meta.shape[-1]))
                zrel = meta_tot.loc[:, ["zrel"]].values
                for z in range(params["N_META"]):
                    # Merge the shots in a single terrain trace
                    xg_meta_tot[z, :xg_meta.shape[-2]
                                ] = xg_meta[z*params["LENGTH_META"]]
                    for p in range(1, params["LENGTH_META"]):
                        xg_meta_tot[z, xg_meta.shape[-2]+16*(p-1):xg_meta.shape[-2]+16*p] = xg_meta[p+z *
                                                                                                    params["LENGTH_META"], -16:]+zrel[p+z*params["LENGTH_META"]]-zrel[z*params["LENGTH_META"]]
            else:
                xg_meta_tot = xg_meta

            if not i:
                XG_SHOTS = np.expand_dims(xg_shots_tot, axis=0)
                XE_SHOTS = np.expand_dims(xe_shots, axis=0)
                XG_META = np.expand_dims(xg_meta_tot, axis=0)
                YE_META = np.expand_dims(ye_meta, axis=0)
                if params["SHOTS_EXTRA_INFO"]:
                    XEI_SHOTS = np.expand_dims(xei_shots, axis=0)
            else:
                XG_SHOTS = np.concatenate(
                    [XG_SHOTS, np.expand_dims(xg_shots_tot, axis=0)], axis=0)
                XE_SHOTS = np.concatenate(
                    [XE_SHOTS, np.expand_dims(xe_shots, axis=0)], axis=0)
                XG_META = np.concatenate(
                    [XG_META, np.expand_dims(xg_meta_tot, axis=0)], axis=0)
                YE_META = np.concatenate(
                    [YE_META, np.expand_dims(ye_meta, axis=0)], axis=0)
                if params["SHOTS_EXTRA_INFO"]:
                    XEI_SHOTS = np.concatenate(
                    [XEI_SHOTS, np.expand_dims(xei_shots, axis=0)], axis=0)

        if params["REMOVE_CENTRAL_BAND"]:
            XG_l = XG_SHOTS[:, :, :, :(
                params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2]
            XG_r = XG_SHOTS[:, :, :, -(params["WHEEL_TRACE_SHAPE"]
                                       [1]-params["REMOVE_CENTRAL_BAND"])//2:]
            XG_SHOTS = np.concatenate([XG_l, XG_r], axis=-1)
            XG_l = XG_META[:, :, :, :(
                params["WHEEL_TRACE_SHAPE"][1]-params["REMOVE_CENTRAL_BAND"])//2]
            XG_r = XG_META[:, :, :, -(params["WHEEL_TRACE_SHAPE"]
                                      [1]-params["REMOVE_CENTRAL_BAND"])//2:]
            XG_META = np.concatenate([XG_l, XG_r], axis=-1)
            
        if params["SHOTS_EXTRA_INFO"]:
            XEI_SHOTS_TOT = np.concatenate([XE_SHOTS, XEI_SHOTS], axis = -1)
        else:
            XEI_SHOTS_TOT = XE_SHOTS
        X = [XG_SHOTS, XEI_SHOTS_TOT, XG_META]
        Y = []
        
        for sh in range(params["N_SHOTS"]):
            if params["MERGED_META_OUTPUT"]:
                for z in range(params["N_META"]):
                    Y.append(
                        np.sum(YE_META[:, z*params["LENGTH_META"]:(z+1)*params["LENGTH_META"]], axis=1))
            else:
                for z in range(params["N_META"]*params["LENGTH_META"]):
                    Y.append(YE_META[:, z])
                
                    
        if self.mode == "prediction":
            return X
        else:
            return X, Y


def main():
    id_files = [".csv"]
    df = pd.DataFrame()
    for i, path_data in enumerate(params["path_data"]):
        files = os.listdir(path_data)
        for file in files:
            if any([p in file for p in id_files]):
                dfi = pd.read_csv(path_data+file)    
                ## Removing some of data:
                # Samples without failures
                try:
                    dfi = dfi[dfi.goal==1]
                except:
                    pass
                # Samples without initial acceleration
                dfi = dfi[dfi.segment!=0]
                try:
                    # Samples without low mean speed
                    dfi = dfi[dfi.mean_speed>0.87]
                    # Samples without low initial speed
                    dfi = dfi[dfi.initial_speed>0.88]
                    
                    dfi["mean_speed_long"] = dfi.mean_speed
                    dfi["initial_speed_long"] = dfi.initial_speed
                    
                except:
                    # Samples without low mean speed
                    dfi = dfi[dfi.mean_speed_long>0.87]
                    # Samples without low initial speed
                    dfi = dfi[dfi.initial_speed_long>0.88]
                if params["REMOVE_ROUGH"]:
                    # Samples without rough pitch/roll variations
                    dfi = dfi.loc[(dfi.var_pitch_est <=params["MAX_VAR_ROUGH"]) | (dfi.var_roll_est <=params["MAX_VAR_ROUGH"])]
                # dfi["std_pitch_est"] = dfi["var_pitch_est"].pow(0.5)
                # dfi["std_roll_est"] = dfi["var_roll_est"].pow(0.5)
                # dfi["curvature"] = dfi["curvature"]*100
                # dfi["curvature_tm1"] = dfi["curvature_tm1"]*100
                
                try:
                    dfi = dfi.drop(columns=['wheel_types'])
                except:
                    pass
                dfi["energy"] = dfi["energy"].clip(lower = 0.0)
                
                dfi["dataset"] = [i]*len(dfi)
                df = pd.concat([df,dfi])
    
    df_sum_indices = pd.DataFrame()
    for i, path_idx in enumerate(params["path_sum_indices"]):         
        dfi = pd.read_csv(path_idx).drop_duplicates()
        dfi["dataset"] = [i]*len(dfi)
        df_sum_indices = pd.concat([df_sum_indices,dfi], ignore_index=True)
    try:
        df = df.drop(columns=['wheel_types'])
    except:
        pass
    df["energy"] = df["energy"].clip(lower = 0.0)
    df["energy"] += params["EPS"]

    for comb in range(n_combs):
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        LOG_NAME = "{}/".format(current_time)

        if not len(IDS_TRAIN):
            # Selection of terrains for training and validation by category
            id_val = []
            id_train = []
            for cat in range(len(CATEGORIES)):
                t_ids = CATEGORIES[cat]
                id_train.extend(random.sample(
                    t_ids, params["TRAINING_DATASETS_PER_CATEGORY"][cat]))
                id_val.extend([t for t in t_ids if t not in id_train])
        else:
            id_train = IDS_TRAIN[comb]
            id_val = []
            for cat in range(len(CATEGORIES)):
                t_ids = CATEGORIES[cat]
                id_val.extend([t for t in t_ids if t not in id_train])
        params["TERRAIN_IDS_TRAIN"] = id_train
        params["TERRAIN_IDS_VAL"] = id_val

        df_train = df[df["terrain_id"].isin(id_train)]
        df_val = df[df["terrain_id"].isin(id_val)]
        if params["TRAIN_PERC"] < 1:
            df_train = df_train.sample(frac=params["TRAIN_PERC"])
            
        df_y = df_train["energy"]
        min_y = df_y.min()
        max_y = df_y.max()
        int_y = df_y.max()-df_y.min()
        params["energy_min"] = min_y
        params["energy_max"] = max_y
        params["energy_int"] = int_y
        
        if params["ENERGY_NORM_TYPE"] == 'standardize':
            df_y = df_train["energy"]
            mean_y = df_y.mean()
            std_y = df_y.std()
            df_train["energy"] = (df_train["energy"]-mean_y)/std_y
            df_val["energy"] = (df_val["energy"]-mean_y)/std_y
            params["energy_mean"] = mean_y
            params["energy_std"] = std_y
            del df_y
        elif params["ENERGY_NORM_TYPE"] == 'normalize':
            df_y = df_train["energy"]
            min_y = df_y.min()
            max_y = df_y.max()
            int_y = df_y.max()-df_y.min()
            df_train["energy"] = (df_train["energy"]-min_y)/int_y
            df_val["energy"] = (df_val["energy"]-min_y)/int_y
            params["energy_min"] = min_y
            params["energy_max"] = max_y
            params["energy_int"] = int_y
            del df_y
        elif params["ENERGY_NORM_TYPE"] == 'standardize_shift':
            df_y = df_train["energy"]
            mean_y = df_y.mean()
            std_y = df_y.std()
            df_train["energy"] = (df_train["energy"]-mean_y)/std_y + mean_y/std_y
            df_val["energy"] = (df_val["energy"]-mean_y)/std_y + mean_y/std_y
            params["energy_mean"] = mean_y
            params["energy_std"] = std_y
            del df_y
            
        if params["SHOTS_EXTRA_INFO"]:
            if params["EXTRA_INFO_NORM_TYPE"] == 'standardize':
                df_ex = df_train[params["SHOTS_EXTRA_INFO"]]
                df_train[params["SHOTS_EXTRA_INFO"]] = (df_train[params["SHOTS_EXTRA_INFO"]]-df_ex.mean())/df_ex.std()
                df_val[params["SHOTS_EXTRA_INFO"]] = (df_val[params["SHOTS_EXTRA_INFO"]]-df_ex.mean())/df_ex.std()
                params["SHOTS_EXTRA_INFO_mean"] = df_ex.mean()
                params["SHOTS_EXTRA_INFO_std"] = df_ex.std()
                del df_ex
            elif params["EXTRA_INFO_NORM_TYPE"] == 'normalize':
                df_ex = df_train[params["SHOTS_EXTRA_INFO"]]
                min_ex = df_ex.min()
                max_ex = df_ex.max()
                int_ex = df_ex.max()-df_ex.min()
                df_train[params["SHOTS_EXTRA_INFO"]] = (df_train[params["SHOTS_EXTRA_INFO"]]-min_ex)/int_ex
                df_val[params["SHOTS_EXTRA_INFO"]] = (df_val[params["SHOTS_EXTRA_INFO"]]-min_ex)/int_ex
                params["extra_info_min"] = min_ex
                params["extra_info_max"] = max_ex
                params["extra_info_int"] = int_ex
                del df_ex

        print()
        print()
        print("Samples: {}".format(len(df)))
        print("Training Samples: {}".format(len(df_train)))
        print("Validation Samples: {}".format(len(df_val)))
        print("Training Terrains {}".format(id_train))
        print("Validation Terrains {}".format(id_val))
        
        if not os.path.exists(params["LOG_DIR"]+LOG_NAME):
            os.makedirs(params["LOG_DIR"]+LOG_NAME)
        file = open(params["LOG_DIR"]+LOG_NAME +
                    "log_params_{}.txt".format(current_time), "w+")
        for key, val in params.items():
            file.write("{}: {}\n".format(key, val))
        file.close()
        

        model = models.get_model(params, summary=True)

        # Initialise data sequences and callbacks
        seq_train = DataSequence(df_train, df_sum_indices, params["BATCH_SIZE"], mode='train')
        seq_val = DataSequence(df_val, df_sum_indices, params["BATCH_SIZE"], mode='validation')

        del df_train, df_val

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=params["MAX_NO_IMPROVEMENT_EPOCHS"]//params["FREQ_VAL"],
                                                   mode='min', verbose=1),
                     keras.callbacks.TensorBoard(log_dir=params["LOG_DIR"] + LOG_NAME,
                                                 update_freq="epoch",
                                                 histogram_freq=0),
                     keras.callbacks.CSVLogger(params["LOG_DIR"] + LOG_NAME + '\\log.csv',
                                               append=True),
                     keras.callbacks.ModelCheckpoint(
            params["LOG_DIR"] + LOG_NAME + 'model_best.hdf5',
            monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')]
        
        
        if params["ESTIMATION_TYPE"] == "deterministic":
            model.compile(optimizer=keras.optimizers.RMSprop(lr=params["LEARNING_RATE"]),
                          loss=params["LOSS"], metrics=[tfa.metrics.RSquare(dtype=tf.float32, y_shape=(1,))] if params["METRICS"] else [])
        elif params["ESTIMATION_TYPE"] == "probability":
            model.compile(optimizer=keras.optimizers.RMSprop(lr=params["LEARNING_RATE"]),
                          loss=lambda y, p_y: -p_y.log_prob(y), metrics=[MSE,R_squared] if params["METRICS"] else [])
        model.fit(seq_train,
                  validation_data=seq_val,
                  validation_freq=params["FREQ_VAL"],
                  use_multiprocessing=bool(max(0, n_workers-1)),
                  workers=n_workers,
                  initial_epoch = params["EPOCHS_FREEZED"],
                  epochs=params["EPOCHS"], callbacks=callbacks)


if __name__ == "__main__":
    main()
