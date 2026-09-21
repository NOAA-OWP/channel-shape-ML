import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class PyTorchTabularWrapper(BaseEstimator, RegressorMixin):
    """Scikit-Learn estimator wrapper around PyTorch Neural Networks."""
    def __init__(self, net_cls=None, net_params=None, dl_params=None, transform_target='log'):
        self.net_cls = net_cls
        self.net_params = net_params or {}
        self.dl_params = dl_params or {}
        self.transform_target = transform_target
        
        self.lr = self.dl_params.get('lr', 0.0005)
        self.weight_decay = self.dl_params.get('weight_decay', 1e-3)
        self.batch_size = self.dl_params.get('batch_size', 256)
        self.max_epochs = self.dl_params.get('max_epochs', 300)
        self.patience = self.dl_params.get('patience', 30)
        self.min_delta = self.dl_params.get('min_delta', 1e-6)
        self.target_min_clip = self.dl_params.get('target_min_clip', 0.01)
        self.target_max_clip = self.dl_params.get('target_max_clip', 0.35)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_in_ = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_rmse_orig': [], 'lr': [], 'best_epoch': 0}

    def _compute_hybrid_loss(self, preds_log, y_log, weights):
        loss_log = (weights * (preds_log - y_log) ** 2).sum() / weights.sum()
        return loss_log

    def fit(self, X, y, sample_weight=None, eval_set=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
            X_arr = X.values
        else:
            X_arr = X

        y_arr = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        w_arr = sample_weight.values if isinstance(sample_weight, (pd.Series, pd.DataFrame)) else sample_weight

        if w_arr is None:
            w_arr = np.ones(len(y_arr), dtype=np.float32)

        X_scaled = self.scaler.fit_transform(X_arr)
        in_features = X_scaled.shape[1]

        self.model = self.net_cls(in_features=in_features, **self.net_params).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)
        w_tensor = torch.tensor(w_arr, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor, w_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            total_weight = 0.0

            for bx, by, bw in loader:
                bx, by, bw = bx.to(self.device), by.to(self.device), bw.to(self.device)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = self._compute_hybrid_loss(preds, by, bw)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * bw.sum().item()
                total_weight += bw.sum().item()

            mean_train_loss = epoch_train_loss / total_weight
            curr_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            if eval_set is not None:
                if len(eval_set) == 3:
                    val_x, val_y, val_w = eval_set
                else:
                    val_x, val_y = eval_set
                    val_w = None

                val_x_arr = val_x.values if isinstance(val_x, (pd.DataFrame, pd.Series)) else val_x
                val_y_arr = val_y.values if isinstance(val_y, (pd.DataFrame, pd.Series)) else val_y
                val_w_arr = val_w.values if isinstance(val_w, (pd.Series, pd.DataFrame)) else val_w

                if val_w_arr is None:
                    val_w_arr = np.ones(len(val_y_arr), dtype=np.float32)

                val_x_scaled = self.scaler.transform(val_x_arr)
                val_x_t = torch.tensor(val_x_scaled, dtype=torch.float32).to(self.device)
                val_y_t = torch.tensor(val_y_arr, dtype=torch.float32).to(self.device)
                val_w_t = torch.tensor(val_w_arr, dtype=torch.float32).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_x_t)
                    val_loss = self._compute_hybrid_loss(val_preds, val_y_t, val_w_t).item()

                val_preds_np = val_preds.cpu().numpy()
                if self.transform_target == 'log':
                    val_preds_orig = np.clip(np.exp(val_preds_np), self.target_min_clip, self.target_max_clip)
                    val_y_orig = np.exp(val_y_arr)
                else:
                    val_preds_orig = val_preds_np
                    val_y_orig = val_y_arr

                val_rmse_orig = np.sqrt(mean_squared_error(val_y_orig, val_preds_orig))

                self.history['train_loss'].append(mean_train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_rmse_orig'].append(val_rmse_orig)
                self.history['lr'].append(curr_lr)

                if val_loss < (best_val_loss - self.min_delta):
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.history['best_epoch'] = best_epoch
        return self

    def predict(self, X, batch_size=32768):
        """Batched PyTorch inference pass preventing RAM/CUDA VRAM memory crashes."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        curr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to(curr_device)

        X_scaled = self.scaler.transform(X_arr)
        num_samples = len(X_scaled)
        preds_list = []

        self.model.eval()
        with torch.inference_mode():
            for i in range(0, num_samples, batch_size):
                batch_x = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32, device=curr_device)
                out = self.model(batch_x)
                preds_list.append(out.cpu().numpy())

        return np.concatenate(preds_list, axis=0) if preds_list else np.array([], dtype=np.float32)