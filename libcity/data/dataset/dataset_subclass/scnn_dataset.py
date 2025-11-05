import os

import pandas as pd

from libcity.data.dataset.traffic_state_datatset import TrafficStateDataset


class SCNNDataset(TrafficStateDataset):
    """Dataset adaptor for SCNN that supports both point-based and grid-based data."""

    def __init__(self, config):
        self.use_row_column = config.get('use_row_column', False)
        self._is_grid = False
        super().__init__(config)

        suffix = 'grid_rc' if self._is_grid and self.use_row_column else (
            'grid' if self._is_grid else 'point')
        self.parameters_str = self.parameters_str + '_' + suffix
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/',
            f'scnn_{self.parameters_str}.npz')

    def _load_geo(self):
        geo_path = os.path.join(self.data_path, self.geo_file + '.geo')
        sample = pd.read_csv(geo_path, nrows=1)
        if {'row_id', 'column_id'}.issubset(sample.columns):
            self._is_grid = True
            TrafficStateDataset._load_grid_geo(self)
        else:
            self._is_grid = False
            TrafficStateDataset._load_geo(self)

    def _load_rel(self):
        if self._is_grid and not os.path.exists(os.path.join(self.data_path, self.rel_file + '.rel')):
            TrafficStateDataset._load_grid_rel(self)
        else:
            TrafficStateDataset._load_rel(self)

    def _load_dyna(self, filename):
        if self._is_grid:
            if self.use_row_column:
                return TrafficStateDataset._load_grid_4d(self, filename)
            return TrafficStateDataset._load_grid_3d(self, filename)
        return TrafficStateDataset._load_dyna_3d(self, filename)

    def _add_external_information(self, df, ext_data=None):
        if self._is_grid and self.use_row_column:
            return TrafficStateDataset._add_external_information_4d(self, df, ext_data)
        return TrafficStateDataset._add_external_information_3d(self, df, ext_data)

    def get_data_feature(self):
        feature = {
            "scaler": self.scaler,
            "adj_mx": self.adj_mx,
            "num_nodes": self.num_nodes,
            "feature_dim": self.feature_dim,
            "ext_dim": self.ext_dim,
            "output_dim": self.output_dim,
            "num_batches": self.num_batches
        }
        if self._is_grid:
            feature.update({
                "len_row": getattr(self, 'len_row', None),
                "len_column": getattr(self, 'len_column', None),
                "grid_use_row_column": self.use_row_column
            })
        return feature


