import torch
import os

torch.backends.cudnn.benchmark =  True

from .forgemaster_base import _Forgemaster

class ForgemasterPull(_Forgemaster):

    def _define_objective(self, net, inputs, labels, args):
        """Implement the closure here."""
        criterion = torch.nn.MSELoss()
        embeddings, outputs = net(inputs)
        refer_emb = torch.load(os.path.join(args.centroid_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pt'), map_location=self.setup['device'])
        new_labels = self._label_map(outputs, labels,refer_emb, criterion, embeddings)
        loss = criterion(embeddings, new_labels)
        loss.backward(retain_graph=self.retain)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        return loss.detach().cpu(), prediction.detach().cpu()


    def _label_map(self, outputs, labels, refer_emb, criterion, embeddings):
        new_labels = torch.stack([self._find_close(refer_emb, labels[i].item(),criterion, embeddings[i]) for i in range(labels.shape[0])])
        return new_labels
    
    def _find_close(self, refer_emb, labels_single, criterion, embeddings_single):
        current_dis = [0 for i in range(len(refer_emb))]
        for i in range(len(refer_emb)):
            current_dis[i] = criterion(refer_emb[i],embeddings_single).item()
        current_dis[labels_single] = 1
        min_val = min(current_dis)
        best_idx = current_dis.index(min_val)
        closest_emb = refer_emb[best_idx]
        return closest_emb
