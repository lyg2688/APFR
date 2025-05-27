import torch
import torch.nn.functional as F
from engine import Engine
from Tranformer import Transformer

class Trans_model(torch.nn.Module):
    def __init__(self, config):
        super(Trans_model, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # self.embedding_ratings = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dim)
        self.affine_output = Transformer(self.latent_dim, self.latent_dim)
        # self.affine_output = torch.nn.Transformer(d_model=32, nhead=8, num_encoder_layers=1,
        #                                           num_decoder_layers=1, dim_feedforward=256, dropout=0.1,
        #                                           activation=F.relu, custom_encoder=None, custom_decoder=None,
        #                                           layer_norm_eps=1e-05, batch_first=False, norm_first=False,
        #                                           device=None,
        #                                           dtype=None)
        # self.to_one = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        item_embedding = self.embedding_item(item_indices).unsqueeze(1)
        # rating_embedding = self.embedding_ratings(ratings.int()).unsqueeze(1)
        # logits = self.affine_output(item_embedding, rating_embedding).squeeze(1)
        logits = self.affine_output(item_embedding).squeeze(1)
        # logits = self.to_one(logits)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class MLPEngine(Engine):

    def __init__(self, config):
        self.model = Trans_model(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
