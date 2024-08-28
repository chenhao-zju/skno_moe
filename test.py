import os
import time
import copy
import argparse
import numpy as np
import torch
import math
import logging
from utils import logging_utils
from utils.YParams import YParams

from utils.data_loader_npyfiles import get_data_loader_npy, FEATURE_DICT, SIZE_DICT
# from model import SphericalFourierNeuralOperatorNet as SKNO
from networks import SKNO
from inference import WeatherForecast

logging_utils.config_logger()

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    # try:
    new_state_dict = {}
    for key, val in checkpoint['model_state'].items():
        name = key[7:]
        # if name != 'ged':
        new_state_dict[name] = val  
    model.load_state_dict(new_state_dict)
    # except:
    #     model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, dataset):
        super(InferenceModule, self).__init__(model, config, dataset)
        # statistic_dir = os.path.join(config["data"]["root_dir"], "statistic")
        # mean = np.load(os.path.join(statistic_dir, "mean.npy"))
        # mean = mean.transpose(1, 2, 3, 0)  # HWFL(1, 1, 5, 13)
        # mean = mean.reshape((1, -1))
        # mean = np.squeeze(mean, axis=0)
        # mean_s = np.load(os.path.join(statistic_dir, "mean_s.npy"))
        # self.mean_all = np.concatenate([mean, mean_s], axis=-1)

        # std = np.load(os.path.join(statistic_dir, "std.npy"))
        # std = std.transpose(1, 2, 3, 0)
        # std = std.reshape((1, -1))
        # self.std = np.squeeze(std, axis=0)
        # self.std_s = np.load(os.path.join(statistic_dir, "std_s.npy"))
        # self.std_all = np.concatenate([self.std, self.std_s], axis=-1)
        self.dataset = dataset

        self.mean = self.dataset.mean_pressure_level
        self.mean_s = self.dataset.mean_surface
        self.mean_all = np.concatenate([self.mean.transpose((1,0)).reshape([-1]), self.mean_s], axis=-1)

        self.std = self.dataset.std_pressure_level
        self.std_s = self.dataset.std_surface
        self.std_all = np.concatenate([self.std.transpose((1,0)).reshape([-1]), self.std_s], axis=-1)
        
        self.feature_dims = config['feature_dims']
        self.use_moe = config['use_moe']
        # self.climate = np.load(os.path.join(statistic_dir, "climate_1.4.npy"))

    def forecast(self, inputs):
        pred_lst = []
        with torch.no_grad():
            for _ in range(self.t_out_test):
                # logging.info(f'inputs shape: {inputs.shape}')
                if self.use_moe:
                    pred, _, _ = self.model(inputs)
                else:
                    pred, _ = self.model(inputs)
                pred_lst.append(pred.to('cpu'))
                # logging.info(f'pred shape: {pred.shape}, pred length: {len(pred_lst)}')
                inputs = pred
        return pred_lst

    # def _get_metrics(self, inputs, labels):
    #     """Get lat_weight_rmse and lat_weight_acc metrics"""
    #     batch_size = inputs.shape[0]
    #     pred = self.forecast(inputs)
    #     # pred = ops.stack(pred, 1).transpose(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
    #     pred = torch.stack(pred, dim=1).permute(0, 1, 3, 4, 2)
    #     # labels = labels.permute(0, 2, 3, 4, 1)  # (B,C,T,H W)->BTHWC
    #     labels = labels.permute(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
    #     self.climates = self.climates.permute(0, 1, 3, 4, 2)

    #     pred = pred.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)
    #     labels = labels.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)
    #     self.climates = self.climates.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)

    #     pred = pred - self.climates
    #     labels = labels - self.climates

    #     # rmse
    #     # error = ops.square(pred - labels).transpose(0, 1, 3, 2).reshape(
    #     #     batch_size, self.t_out_test * self.feature_dims, -1)
    #     # weight = ms.Tensor(self._calculate_lat_weight().reshape(-1, 1))
    #     # lat_weight_rmse_step = ops.matmul(error, weight).sum(axis=0)
    #     # lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test,
    #     #                                           self.feature_dims).transpose(1, 0).asnumpy()
    #     error = torch.square(pred - labels).permute(0, 1, 3, 2).reshape(
    #                         batch_size, self.t_out_test * self.feature_dims, -1)
    #     weight = torch.tensor(self._calculate_lat_weight().reshape(-1, 1))
    #     lat_weight_rmse_step = (error @ weight).sum(axis=0)
    #     lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test,
    #                                                         self.feature_dims).permute(1, 0).numpy()

    #     # acc
    #     # pred = pred * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
    #     # labels = labels * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
    #     # pred = pred - ms.Tensor(self.climate, ms.float32)
    #     # labels = labels - ms.Tensor(self.climate, ms.float32)
    #     pred = pred * torch.tensor(self.std_all, dtype=torch.float32) + torch.tensor(self.mean_all, dtype=torch.float32)
    #     labels = labels * torch.tensor(self.std_all, dtype=torch.float32) + torch.tensor(self.mean_all, dtype=torch.float32)
    #     # pred = pred - torch.tensor(self.climate, dtype=torch.float32)
    #     # labels = labels - torch.tensor(self.climate, dtype=torch.float32)

    #     acc_numerator = pred * labels
    #     acc_numerator = acc_numerator.permute(0, 1, 3, 2).reshape(
    #         batch_size, self.t_out_test * self.feature_dims, -1)
    #     # acc_numerator = ops.matmul(acc_numerator, weight)
    #     acc_numerator = acc_numerator @ weight

    #     # pred_square = ops.square(pred).transpose(0, 1, 3, 2).reshape(
    #     #     batch_size, self.t_out_test * self.feature_dims, -1)
    #     # label_square = ops.square(labels).transpose(0, 1, 3, 2).reshape(
    #     #     batch_size, self.t_out_test * self.feature_dims, -1)
    #     pred_square = torch.square(pred).permute(0, 1, 3, 2).reshape(
    #         batch_size, self.t_out_test * self.feature_dims, -1)
    #     label_square = torch.square(labels).permute(0, 1, 3, 2).reshape(
    #         batch_size, self.t_out_test * self.feature_dims, -1)

    #     # acc_denominator = ops.sqrt(ops.matmul(pred_square, weight) * ops.matmul(label_square, weight))
    #     # lat_weight_acc = np.divide(acc_numerator.asnumpy(), acc_denominator.asnumpy())
    #     acc_denominator = torch.sqrt((pred_square @ weight) * (label_square @ weight))
    #     lat_weight_acc = np.divide(acc_numerator.numpy(), acc_denominator.numpy())
    #     lat_weight_acc_step = lat_weight_acc.sum(axis=0).reshape(self.t_out_test,
    #                                                              self.feature_dims).transpose(1, 0)
    #     return lat_weight_rmse_step, lat_weight_acc_step

    # def _calculate_lat_weight(self):
    #     lat_t = np.arange(0, self.h_size)
    #     s = np.sum(np.cos(3.1416 / 180. * self._lat(lat_t)))
    #     weight = self._latitude_weighting_factor(lat_t, s)
    #     grid_lat_weight = np.repeat(weight, self.w_size, axis=0)
    #     grid_lat_weight = grid_lat_weight.reshape(-1)
    #     return grid_lat_weight.astype(np.float32)

    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        batch_size = inputs.shape[0]
        pred = self.forecast(inputs)
        # pred = torch.stack(pred, axis=0).asnumpy()
        # labels = labels.asnumpy()

        pred = torch.stack(pred, dim=1).permute(0, 1, 3, 4, 2)
        # labels = labels.permute(0, 2, 3, 4, 1)  # (B,C,T,H W)->BTHWC
        labels = labels.permute(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
        self.climates = self.climates.permute(0, 1, 3, 4, 2)

        pred = pred.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()
        labels = labels.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()
        self.climates = self.climates.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()

        # pred = pred - self.climates
        # labels = labels - self.climates

        lat_weight_rmse_step = self._calculate_lat_weighted_error(labels, pred).transpose()
        lat_weight_acc = self._calculate_lat_weighted_acc(labels, pred).transpose()

        return lat_weight_rmse_step, lat_weight_acc

    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        # self.h_size * np.cos(PI / 180. * self._lat(j)) / s
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _calculate_lat_weighted_error(self, label, prediction):
        """calculate latitude weighted error"""
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1, 1)
        error = np.square(label - prediction) # the index 0 of label shape is batch_size
        # logging.info(f'error shape: {error.shape}, grid_node_weight shape: {grid_node_weight.shape}')
        lat_weight_error = np.sum(error * grid_node_weight, axis=2)
        lat_weight_error = np.sum(lat_weight_error, axis=0)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """calculate latitude weighted acc"""
        # prediction = prediction * self.std_all.reshape((1, 1, -1)) + self.mean_all.reshape((1, 1, -1))
        # label = label * self.std_all.reshape((1, 1, 1, -1)) + self.mean_all.reshape((1, 1, 1, -1))
        prediction = prediction - self.climates
        label = label - self.climates

        prediction = prediction * self.std_all + self.mean_all
        label = label * self.std_all + self.mean_all
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        acc_numerator = np.sum(prediction * label * grid_node_weight, axis=2)
        # acc_denominator = np.sqrt(np.sum(prediction ** 2 * grid_node_weight,
        #                                  axis=2) * np.sum(label ** 2 * grid_node_weight, axis=2))
        acc_denominator = np.sqrt(np.sum(grid_node_weight * prediction ** 2,
                                         axis=2) * np.sum(grid_node_weight * label ** 2, axis=2))
        # acc_numerator = np.sum(acc_numerator, axis=0)
        # acc_denominator = np.sum(acc_denominator, axis=0)

        try:
            # acc = acc_numerator / acc_denominator
            acc = np.divide(acc_numerator, acc_denominator)
            acc = np.sum(acc, axis=0)
        except ZeroDivisionError as e:
            print(repr(e))
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    # params['interp'] = args.interp
    params['global_batch_size'] = params.batch_size
    # params['global_batch_size'] = 32

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    # vis = args.vis

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # model = SKNO(img_size = (params.h_size, params.w_size),
    #             in_chans = params.feature_dims,
    #             out_chans = params.feature_dims,
    #             embed_dim = 768,
    #             num_layers = 16).to(device) 
    model = SKNO(params).to(device)

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    test_data_loader, test_dataset = get_data_loader_npy(params, False, run_mode='test')

    start_time = time.time()
    inference_module = InferenceModule(model, params, test_dataset)
    inference_module.eval(test_data_loader)
    
    logging.info(f"End-to-End total time: {time.time() - start_time} s")
