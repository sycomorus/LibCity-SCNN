import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

epsilon = 0.01


def SeasonalNorm(x, period_length):
    b, c, n, t = x.shape
    x_period = torch.split(x, split_size_or_sections=period_length, dim=-1)
    x_period = torch.stack(x_period, -2)

    mean = x_period.mean(3)
    var = (x_period ** 2).mean(3) - mean ** 2 + 0.00001

    mean = mean.repeat(1, 1, 1, t // period_length)
    var = var.repeat(1, 1, 1, t // period_length)

    mean = F.pad(mean.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='circular', pad=(t % period_length, 0)).reshape(b, c, n, -1)
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5


class AdaSpatialNorm(nn.Module):
    def __init__(self, embedding_dim, num_nodes, device):
        super(AdaSpatialNorm, self).__init__()
        self.node_embedding = nn.Parameter(torch.zeros(num_nodes, embedding_dim))
        self.device = device

    def forward(self, x):
        b, c, n, t = x.shape

        adj_mat = torch.matmul(self.node_embedding, self.node_embedding.T)
        adj_mat = adj_mat - 10 * torch.eye(n, device=x.device)
        adj_mat = torch.softmax(adj_mat, dim=-1)

        adj_mat = adj_mat.unsqueeze(0)
        x_f = x.permute(0, 3, 2, 1).reshape(b * t, -1, c)

        mean_f = torch.matmul(adj_mat, x_f)
        var_f = torch.matmul(adj_mat, x_f ** 2) - mean_f ** 2 + 0.00001

        mean = mean_f.view(b, t, n, c).permute(0, 3, 2, 1)
        var = var_f.view(b, t, n, c).permute(0, 3, 2, 1)

        out = (x - mean) / (var + epsilon) ** 0.5

        return out, mean, var ** 0.5


def PeriodNorm(x, period_len):
    b, c, n, t = x.shape
    x_patch = [x[..., period_len-1-i:-i+t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    var = (x_patch ** 2).mean(4) - mean ** 2 + 0.00001
    mean = F.pad(mean.reshape(b * c, n, -1), mode='replicate', pad=(period_len-1, 0)).reshape(b, c, n, -1)
    var = F.pad(var.reshape(b * c, n, -1), mode='replicate', pad=(period_len-1, 0)).reshape(b, c, n, -1)
    out = (x - mean) / (var + epsilon) ** 0.5

    return out, mean, var ** 0.5


class ResidualExtrapolate(nn.Module):
    def __init__(self, d_model, input_len, output_len):
        super(ResidualExtrapolate, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.regreesor = nn.Conv2d(in_channels=d_model, out_channels=d_model * output_len, kernel_size=(1, input_len))

    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.regreesor(x[..., -self.input_len:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)

        return x_proj


def SeasonalExtrapolate(x, cycle_len, pred_len, cycle_num, device):
    weight = torch.zeros(pred_len // cycle_len + 1, cycle_num, device=device)
    weight = torch.softmax(weight, dim=-1)
    b, c, n, t = x.shape
    x_cycle = torch.split(x, split_size_or_sections=cycle_len, dim=-1)
    x_cycle = torch.stack(x_cycle, -1)
    proj_cycle = torch.matmul(weight, x_cycle.permute(0, 2, 3, 4, 1))
    x_proj = torch.cat([x_cycle.permute(0, 2, 3, 4, 1), proj_cycle], dim=-2).permute(0, 4, 1, 3, 2).reshape(b, c, n, -1)[..., : t + pred_len]

    return x_proj


def ConstantExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='replicate', pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj


class EncoderLayer(nn.Module):
    def __init__(self, d_model, seq_len, pred_len, cycle_len, short_period_len, series_num, kernel_size, device, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.kernel_size = kernel_size
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        self.device = device

        self.spatial_norm = AdaSpatialNorm(d_model, series_num, device)
        self.residual_extrapolate_1 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_2 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_3 = ResidualExtrapolate(d_model, short_period_len, pred_len)
        self.residual_extrapolate_4 = ResidualExtrapolate(d_model, short_period_len, pred_len)

        self.conv_1 = nn.Conv2d(in_channels=13 * d_model, out_channels=d_model, kernel_size=(1, kernel_size), dilation=1)
        self.conv_2 = nn.Conv2d(in_channels=13 * d_model, out_channels=d_model, kernel_size=(1, kernel_size), dilation=1)

        self.skip_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.scale_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.residual_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        b, c, n, t = x.shape
        residual = x
        xs = []

        x_proj = ConstantExtrapolate(x, self.pred_len)
        xs.append(x_proj)

        x, long_term_mean, long_term_std = PeriodNorm(x, self.seq_len)
        x_proj = self.residual_extrapolate_1(x)
        long_term_mean_proj, long_term_std_proj = ConstantExtrapolate(long_term_mean, self.pred_len), ConstantExtrapolate(long_term_std, self.pred_len)
        xs.extend([x_proj, long_term_mean_proj, long_term_std_proj])

        x, season_mean, season_std = SeasonalNorm(x, self.cycle_len)
        x_proj = self.residual_extrapolate_2(x)
        season_mean_proj, season_std_proj = SeasonalExtrapolate(season_mean, self.cycle_len, self.pred_len, self.seq_len // self.cycle_len, x.device), SeasonalExtrapolate(season_std, self.cycle_len, self.pred_len, self.seq_len // self.cycle_len, x.device)
        xs.extend([x_proj, season_mean_proj, season_std_proj])

        x, short_term_mean, short_term_std = PeriodNorm(x, self.short_period_len)
        x_proj = self.residual_extrapolate_3(x)
        short_term_mean_proj, short_term_std_proj = ConstantExtrapolate(short_term_mean, self.pred_len), ConstantExtrapolate(short_term_std, self.pred_len)
        xs.extend([x_proj, short_term_mean_proj, short_term_std_proj])

        x, spatial_mean, spatial_std = self.spatial_norm(x)
        x_proj = self.residual_extrapolate_4(x)
        spatial_mean_proj, spatial_std_proj = ConstantExtrapolate(spatial_mean, self.pred_len), ConstantExtrapolate(spatial_std, self.pred_len)
        xs.extend([x_proj, spatial_mean_proj, spatial_std_proj])

        x = torch.cat(xs, dim=1)
        x = F.pad(x, mode='constant', pad=(self.kernel_size-1, 0))

        x_1 = torch.tanh(self.conv_1(x))
        x_2 = torch.sigmoid(self.conv_2(x))

        x_z = (x_1 * x_2)[..., :-self.pred_len]
        pred_z = (x_1 * x_2)[..., -self.pred_len:]
        s = self.skip_conv(pred_z)
        x_z = self.residual_conv(x_z)

        return x_z, s


class SCNN(AbstractTrafficStateModel):
    """
    SCNN: Spatial Convolutional Neural Network for Time Series Forecasting
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.len_row = data_feature.get('len_row')
        self.len_column = data_feature.get('len_column')
        self.grid_use_row_column = data_feature.get('grid_use_row_column', False)
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 12)
        self.device = config.get('device', torch.device('cpu'))
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        # Model hyperparameters
        self.d_model = config.get('d_model', 8)
        self.e_layers = config.get('e_layers', 2)
        self.kernel_size = config.get('kernel_size', 2)
        self.short_period_len = config.get('short_period_len', 8)
        self.cycle_len = config.get('cycle_len', 24)

        # Build model
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=self.d_model,
                                    kernel_size=1)
        self.enc_layers = nn.ModuleList()
        for i in range(self.e_layers):
            self.enc_layers.append(EncoderLayer(
                self.d_model, self.input_window, self.output_window,
                self.cycle_len, self.short_period_len, self.num_nodes,
                self.kernel_size, self.device, dropout=0.1))
        self.end_conv = nn.Conv1d(in_channels=self.output_window * self.d_model,
                                  out_channels=self.output_window,
                                  groups=self.output_window,
                                  kernel_size=1,
                                  bias=True)

        self._logger.info('SCNN model initialized with num_nodes={}, input_window={}, output_window={}'.format(
            self.num_nodes, self.input_window, self.output_window))

    def forward(self, batch):
        """
        Forward pass of SCNN model

        Args:
            batch (Batch): batch data with 'X' key, shape [B, L, N, F]

        Returns:
            torch.Tensor: prediction, shape [B, T, N, output_dim]
        """
        # Extract input: [B, L, ..., F] -> [B, L, N, output_dim]
        x_enc = batch['X'][..., :self.output_dim]
        grid_shape = None
        if x_enc.dim() == 5:
            b_in, l_in, r_in, c_in, f_in = x_enc.shape
            grid_shape = (r_in, c_in)
            x_enc = x_enc.reshape(b_in, l_in, r_in * c_in, f_in)

        # Process each output dimension channel separately
        predictions = []
        for dim_idx in range(self.output_dim):
            # Extract single channel: [B, L, N, output_dim] -> [B, L, N]
            x_enc_single = x_enc[..., dim_idx]  # [B, L, N]

            # Normalize input (similar to original SCNN)
            means = x_enc_single.mean(1, keepdim=True).detach()  # [B, 1, N]
            x_enc_norm = x_enc_single - means
            stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, N]
            x_enc_norm /= stdev

            # Reshape for SCNN: [B, L, N] -> [B, 1, N, L]
            input_tensor = x_enc_norm.permute(0, 2, 1).unsqueeze(1)  # [B, 1, N, L]
            x = self.start_conv(input_tensor)  # [B, d_model, N, L]

            b, c, n, L = x.shape

            out = 0
            s = 0

            # Encoder layers
            for i in range(self.e_layers):
                residual = x
                x, s = self.enc_layers[i](x)
                x = x + residual
                out = s + out

            # Output projection: [B, T, d_model, N] -> [B, T, N]
            out = out.permute(0, 3, 1, 2).reshape(b, -1, n)  # [B, T*d_model, N]
            dec_out = self.end_conv(out)  # [B, T, N]

            # Denormalize
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_window, 1))  # [B, T, N]
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_window, 1))  # [B, T, N]

            predictions.append(dec_out)

        # Stack predictions: [B, T, N, output_dim]
        dec_out = torch.stack(predictions, dim=-1)  # [B, T, N, output_dim]

        if grid_shape is not None:
            dec_out = dec_out.reshape(
                dec_out.shape[0],
                dec_out.shape[1],
                grid_shape[0],
                grid_shape[1],
                self.output_dim
            )

        return dec_out

    def predict(self, batch):
        """
        Predict method for LibCity interface

        Args:
            batch (Batch): batch data

        Returns:
            torch.Tensor: prediction, shape [B, T, N, output_dim]
        """
        return self.forward(batch)

    def calculate_loss(self, batch):
        """
        Calculate loss for training

        Args:
            batch (Batch): batch data

        Returns:
            torch.Tensor: loss value
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

