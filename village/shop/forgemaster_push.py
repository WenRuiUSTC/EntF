import torch
import os

torch.backends.cudnn.benchmark =  True

from .forgemaster_base import _Forgemaster

class ForgemasterPush(_Forgemaster):

    def _define_objective(self, net, inputs, labels, args):
        """Implement the closure here."""
        criterion = torch.nn.MSELoss()
        embeddings, outputs = net(inputs)
        refer_emb = torch.load(os.path.join(args.centroid_path,args.dataset+'_eps_'+str(int(args.robust_eps))+'.pt'), map_location=self.setup['device'])
        new_labels = self._label_map(outputs, labels,refer_emb)
        loss = -criterion(embeddings, new_labels)
        loss.backward(retain_graph=self.retain)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        return loss.detach().cpu(), prediction.detach().cpu()


    def _label_map(self, outputs, labels, refer_emb):
        new_labels = torch.stack([refer_emb[labels[i].item()] for i in range(labels.shape[0])])
        return new_labels
