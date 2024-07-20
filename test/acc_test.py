import torch
import math
import numpy as np

class acc_test:
    def __init__(self) -> None:
        self.climates = torch.randn(2, 2, 69, 128, 256)
        self.t_out_test = 2
        self.h_size = 128
        self.w_size = 256
        self.feature_dims = 69
        self.std_all = np.random.rand(69)
        self.mean_all = np.random.rand(69)

    def _lat(self, j):
        return 90. - j * 180. / float(self.h_size - 1)

    def _latitude_weighting_factor(self, j, s):
        return self.h_size * np.cos(math.pi / 180. * self._lat(j)) / s

    def _get_metrics1(self, pred, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        batch_size = pred.shape[0]
        # pred = ops.stack(pred, 1).transpose(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
        pred = pred.permute(0, 1, 3, 4, 2)
        # labels = labels.permute(0, 2, 3, 4, 1)  # (B,C,T,H W)->BTHWC
        labels = labels.permute(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
        self.climates = self.climates.permute(0, 1, 3, 4, 2)

        pred = pred.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)
        labels = labels.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)
        self.climates = self.climates.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)

        pred = pred - self.climates
        labels = labels - self.climates

        # rmse
        # error = ops.square(pred - labels).transpose(0, 1, 3, 2).reshape(
        #     batch_size, self.t_out_test * self.feature_dims, -1)
        # weight = ms.Tensor(self._calculate_lat_weight().reshape(-1, 1))
        # lat_weight_rmse_step = ops.matmul(error, weight).sum(axis=0)
        # lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test,
        #                                           self.feature_dims).transpose(1, 0).asnumpy()
        error = torch.square(pred - labels).permute(0, 1, 3, 2).reshape(
                            batch_size, self.t_out_test * self.feature_dims, -1)
        weight = torch.tensor(self._calculate_lat_weight().reshape(-1, 1))
        print('weight1: ', weight[0, 0], weight[300, 0])

        lat_weight_rmse_step = (error @ weight).sum(axis=0)
        lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test,
                                                            self.feature_dims).permute(1, 0).numpy()

        # acc
        # pred = pred * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
        # labels = labels * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
        # pred = pred - ms.Tensor(self.climate, ms.float32)
        # labels = labels - ms.Tensor(self.climate, ms.float32)
        pred = pred * torch.tensor(self.std_all, dtype=torch.float32) + torch.tensor(self.mean_all, dtype=torch.float32)
        labels = labels * torch.tensor(self.std_all, dtype=torch.float32) + torch.tensor(self.mean_all, dtype=torch.float32)
        # pred = pred - torch.tensor(self.climate, dtype=torch.float32)
        # labels = labels - torch.tensor(self.climate, dtype=torch.float32)

        acc_numerator = pred * labels
        acc_numerator = acc_numerator.permute(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)
        # acc_numerator = ops.matmul(acc_numerator, weight)
        acc_numerator = acc_numerator @ weight
        print('acc_numerator1:', acc_numerator[0, :10, 0])

        # pred_square = ops.square(pred).transpose(0, 1, 3, 2).reshape(
        #     batch_size, self.t_out_test * self.feature_dims, -1)
        # label_square = ops.square(labels).transpose(0, 1, 3, 2).reshape(
        #     batch_size, self.t_out_test * self.feature_dims, -1)
        pred_square = torch.square(pred).permute(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)
        label_square = torch.square(labels).permute(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)

        # acc_denominator = ops.sqrt(ops.matmul(pred_square, weight) * ops.matmul(label_square, weight))
        # lat_weight_acc = np.divide(acc_numerator.asnumpy(), acc_denominator.asnumpy())
        acc_denominator = torch.sqrt((pred_square @ weight) * (label_square @ weight))
        print('acc_denominator1:', acc_denominator[0, :10, 0])

        lat_weight_acc = np.divide(acc_numerator.numpy(), acc_denominator.numpy())
        print('lat_weight_acc1: ', lat_weight_acc[0, :10])

        lat_weight_acc_step = lat_weight_acc.sum(axis=0).reshape(self.t_out_test,
                                                                    self.feature_dims).transpose(1, 0)
        return lat_weight_rmse_step, lat_weight_acc_step

    def _calculate_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(3.1416 / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_lat_weight = np.repeat(weight, self.w_size, axis=0)
        grid_lat_weight = grid_lat_weight.reshape(-1)
        return grid_lat_weight.astype(np.float32)

    def _get_metrics2(self, pred, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        self.batch_size = pred.shape[0]
        # pred = torch.stack(pred, axis=0).asnumpy()
        # labels = labels.asnumpy()

        pred = pred.permute(0, 1, 3, 4, 2)
        # labels = labels.permute(0, 2, 3, 4, 1)  # (B,C,T,H W)->BTHWC
        labels = labels.permute(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
        # self.climates = self.climates.permute(0, 1, 3, 4, 2)

        pred = pred.reshape(self.batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()
        labels = labels.reshape(self.batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()
        # self.climates = self.climates.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims).numpy()
        self.climates = self.climates.numpy()

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
        print('weight2: ', grid_node_weight[0, 0, 0], grid_node_weight[0, 300, 0])
        acc_numerator = np.sum(prediction * label * grid_node_weight, axis=2)
        # acc_denominator = np.sqrt(np.sum(prediction ** 2 * grid_node_weight,
        #                                  axis=2) * np.sum(label ** 2 * grid_node_weight, axis=2))
        acc_denominator = np.sqrt(np.sum(grid_node_weight * prediction ** 2,
                                            axis=2) * np.sum(grid_node_weight * label ** 2, axis=2))
        print('acc_numerator2:', acc_numerator.reshape([self.batch_size, -1])[0, :10])
        print('acc_denominator2:', acc_denominator.reshape([self.batch_size, -1])[0, :10])

        # acc_numerator = np.sum(acc_numerator, axis=0)
        # acc_denominator = np.sum(acc_denominator, axis=0)

        try:
            # acc = acc_numerator / acc_denominator
            acc = np.divide(acc_numerator, acc_denominator)
            print('acc2: ', acc[0, :10])
            acc = np.sum(acc, axis=0)

        except ZeroDivisionError as e:
            print(repr(e))
        return acc


if __name__ == '__main__':
    pred = torch.randn(2, 2, 69, 128, 256)
    labels = torch.randn(2, 2, 69, 128, 256)

    acc_test = acc_test()

    rmse1, acc1 = acc_test._get_metrics1(pred, labels)
    rmse2, acc2 = acc_test._get_metrics2(pred, labels)

    print(rmse1[:7,0], rmse2[:7,0])
    print(acc1[:7,0], acc2[:7,0])