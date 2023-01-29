import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
from mtpgr.new_features.adjacency_matrix_v2 import AdjacencyMatrixV2
from mtpgr.new_features.isolated_dataset import IsolatedDataset
from mtpgr.new_features.model_manager import ModelManager
from mtpgr.config.defaults import get_cfg_defaults
from mtpgr.new_features.vibe_frame_dataset import VibeFrameDataset

# Confusion matrix
class PlotConfusionMatrix:

    def __init__(self) -> None:
        self.pkl_path = Path('eval_result.pkl')

    def compute_confusion_matrix(self, y_true, y_pred, save_folder:Path=None):
        cm = confusion_matrix(y_true, y_pred, )
        # np.set_printoptions(precision=2)
        # print('Confusion matrix, without normalization')
        # print(cm)
        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        # cm = cm[1:, 1:]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print('Normalized confusion matrix')
        # print(cm_normalized)

        if save_folder is not None:
            np.savetxt(save_folder / "cm.txt", cm, fmt='%-6d')
            np.savetxt(save_folder / "cm_norm.txt", cm_normalized, fmt='%.2f')
        
        # plot_confusion_matrix(cm)
        # plt.savefig(save_folder / "cm_norm.pdf")
        return cm_normalized


    def plot_confusion_matrix(self, cm, cmap=plt.cm.Greys):
        fig, ax = plt.subplots(1,1, figsize=(20, 20))
        font_size = 8
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        for i in range(cm.shape[0]): 
            for j in range(cm.shape[1]): 
                number = cm[i, j]
                if number > 0.05:
                    text = "{:0.2f}".format(number)
                    color = 'white' if number > 0.5 else 'black'
                    plt.text(x=j, y=i,s=text, va='center', ha='center', fontsize=font_size, color=color)
        # plt.title(title)
        # plt.colorbar
        tick_marks = np.arange(len(range(0, 33)))
        ax.set_xticks(tick_marks, range(0, 33), fontsize=font_size, rotation=45)
        ax.set_yticks(tick_marks, range(0, 33), fontsize=font_size)
        
        ax.set_ylabel('Ground Truth', fontsize=font_size)
        ax.set_xlabel('Prediction', fontsize=font_size)
        plt.tight_layout()
        plt.show()
    
    def plot(self):
        with self.pkl_path.open('rb') as f:
            result = pickle.load(f)
        matrix = self.compute_confusion_matrix(result['y_true'], result['y_pred'])
        self.plot_confusion_matrix(matrix)    


# Connection strength
class PlotConnectionStrength:

    def __init__(self) -> None:

        cfg = get_cfg_defaults()

        self.test_dataset = IsolatedDataset(cfg, 'test', 'global', do_augment=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,)

        self.mm = ModelManager(cfg)
        self.model = self.mm.model
        self.mm.load_ckpt()

        self.adj = AdjacencyMatrixV2.from_config(cfg).get_adjacency().cuda()  # PHW

    def connection_strength(self):
        edge_importance = self.model.edge_importance
        A_LPHW = torch.stack(list(edge_importance))  # layer, partitioning, h, w
        A_PHW = torch.mean(A_LPHW, dim=0)
        # A_PHW = A_LPHW[5]

        A_HW = torch.sum(A_PHW * self.adj, dim=0)

        adj_HW, _ = torch.max(self.adj, dim=0)

        batch_data = iter(self.test_loader).next()
        x, y_true = self.mm.prepare_data(batch_data)  # x: NCTV
        time = 10
        x = x[0, :, time, :]  # CV
        x = x.cpu().numpy()

        # Adj matrix to edges
        edge_upper = []  # (arr_CV, strength)
        edge_lower = []
        H,W = A_HW.size()
        for i in range(H):
            for j in range(W):
                if adj_HW[i,j] == 0:  # Not connected
                    continue
                
                strength = A_HW[i,j]
                ab = (i,j)
                
                if i < j:  # connection direction
                    edge_upper.append((x[:, ab], strength))
                elif i > j:
                    edge_lower.append((x[:, ab], strength))

        return x, A_HW.cpu().numpy(), edge_upper, edge_lower
        pass
    
    def plot_connections(self, j3d, A_HW, edge_upper, edge_lower):

        fig = plt.figure(figsize=(20, 10))
        
        # ax.set_xlim(0, 2)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-0.75, 1.25)
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(A_HW, cmap=plt.cm.Greys)
        ax.set_xlabel('vertex number')
        ax.set_ylabel('vertex number')

        for i, edges in enumerate((edge_upper, edge_lower,)):
            # Points
            ax = fig.add_subplot(1, 3, i+2, projection='3d')
            ax.scatter3D(j3d[0], j3d[1], j3d[2])
            # Connections
            for arr_CV, strength in edges:
                linewidth = strength*7
                ax.plot3D(*arr_CV, lw=linewidth)
            # ax.axis('off')
            ax.view_init(110, 90)
        
        plt.tight_layout()
        plt.show()
        
    
    def plot(self):
        with torch.no_grad():
            args = self.connection_strength()
        self.plot_connections(*args)
        pass

if __name__ == '__main__':
    # PlotConfusionMatrix().plot()
    PlotConnectionStrength().plot()