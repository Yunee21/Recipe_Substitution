import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, AttentionalAggregation


class HeteroEncoderHAN(nn.Module):
    def __init__(self, ingre_fv_dim, direc_fv_dim, hidden_dim, latent_dim, metadata, heads=1, dropout=0.0):
        super().__init__()


        layer_mode = ['1layer', '2layer', '3layer']
        output_mode = ['node', 'graph', 'together']

        self.layer_mode = layer_mode[1]
        self.output_mode = output_mode[0]

        self.metadata = metadata

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by number of heads."

        # Define input and intermediate channel dictionaries
        in_channels = {'ingredient': ingre_fv_dim, 'direction': direc_fv_dim}
        inter_channels = {ntype: hidden_dim for ntype in in_channels}

        # HANConv Layers
        self.han1 = HANConv(in_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        if layer_mode in ['2layer', '3layer']:
            self.han2 = HANConv(inter_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        if layer_mode == '3layer':
            self.han3 = HANConv(inter_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        '''
        # Graph-level pooling
        self.pool_ingre = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        )
        self.pool_direc = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        )
        '''

        # GNN 1-layer
        in_channels = {'ingredient': ingre_fv_dim, 'direction': direc_fv_dim}

        self.han1 = HANConv(in_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        # GNN 2-layer
        in_channels = {'ingredient': ingre_fv_dim, 'direction': direc_fv_dim}
        inter_channels = {ntype: hidden_dim for ntype in in_channels}

        self.han1 = HANConv(in_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)
        self.han2 = HANConv(inter_channels, hidden_dim, heads=heads, dropout=dropout, metadata=metadata)

        if self.output_mode == 'graph' :
            # graph-level embedding (pooled ingredient + direction)
            self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

        # node-level embedding
        if self.output_mode == 'node':
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # graph-level + node-level embedding
        if self.output_mode == 'together':
            self.fc_mu = nn.Linear(3 * hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(3 * hidden_dim, latent_dim)

        # Deep Layer
        self.mlp_mu = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, data):
        x_dct = {'ingredient': data['ingredient'].x, 'direction': data['direction'].x}

        edge_index_dct = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}

        # GNN 1-layer
        x2 = self.han1(x_dct, edge_index_dct)


        # GNN 2-layer
        x1 = self.han1(x_dct, edge_index_dct)
        x2 = self.han2(x1, edge_index_dct)


        # masking
        mask = data['ingredient'].mask
        assert mask.sum() > 0, "No masked ingredient nodes found in the batch!"
        masked_node_idx = mask.nonzero(as_tuple=True)[0]  # shape: [k]

        if len(masked_node_idx) == 1:
            masked_node_emb = x2['ingredient'][masked_node_idx[0]].unsqueeze(0)  # shape: [1, hidden_dim]
        else:
            masked_node_emb = x2['ingredient'][masked_node_idx]  # shape: [k, hidden_dim]


        # Transfer the neighbors information to decoder (1-hop neighbors from 'co_occurs_with')
        adj = edge_index_dct[('ingredient', 'co_occurs_with', 'ingredient')]

        neighbors = []
        for idx in masked_node_idx:
            neigh = adj[1][adj[0] == idx]
            if neigh.numel() > 0:  # 빈 텐서는 넣지 않음
                neighbors.append(neigh)

        if len(neighbors) > 0:
            neighbors = torch.cat(neighbors)
        else:
            neighbors = torch.empty(0, dtype=torch.long, device=adj.device)

        if neighbors.numel() > 0:
            context = x2['ingredient'][neighbors].mean(dim=0, keepdim=True)
        else:
            context = torch.zeros(1, x2['ingredient'].size(1), device=x2['ingredient'].device)

        '''
        # graph-level ([k, 2*hidden_dim])
        ingre_pool = x2['ingredient'].mean(dim=0, keepdim=True)
        direc_pool = x2['direction'].mean(dim=0, keepdim=True)

        graph_repr = torch.cat([ingre_pool, direc_pool], dim=1)

        mu = self.fc_mu(graph_repr)
        logvar = self.fc_logvar(graph_repr)
        '''

        # node-level ([k, hidden_dim])
        mu = self.fc_mu(masked_node_emb)
        logvar = self.fc_logvar(masked_node_emb)

        '''
        # node-graph-level ([k, hidden_dim + 2*hidden_dim])
        graph_repr = graph_repr.expand(masked_node_emb.size(0), -1)
        encoder_input = torch.cat([masked_node_emb, graph_repr], dim=1)

        mu = self.fc_mu(encoder_input)
        logvar = self.fc_mu(encoder_input)
        '''

        return mu, logvar, context


class HeteroDecoder(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dim, out_dim):
        super().__init__()
        '''
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + context_dim, hidden_dim),
            #nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(context_dim + hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(context_dim + hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(context_dim, hidden_dim),
            #nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, out_dim),
            #nn.ReLU()  # It causes non-negative value
            #nn.Tanh()
        )
        '''
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(latent_dim + context_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.act4 = nn.LeakyReLU(0.1)

        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)
        self.act5 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.act3 = nn.LeakyReLU(0.1)

        self.out_layer = nn.Linear(hidden_dim, out_dim)


    def forward(self, z_and_context):
        #return self.mlp(z_and_context)
        # z_and_context: [batch, latent + context]
        zc = z_and_context

        # split context for reuse
        z = zc[:, :self.latent_dim]
        context = zc[:, -self.context_dim:]  # [latent_dim:]

        noisy_context = context + 0.05 * torch.randn_like(context)

        h = self.fc1(zc)
        h = self.act1(h)
        h = self.norm1(h)

        '''
        #h = torch.cat([h, context], dim=-1)
        h = self.fc2(h)
        h = self.act2(h)
        h = self.norm2(h)

        h = self.fc4(h)
        h = self.act4(h)
        h = self.norm4(h)

        h = self.fc5(h)
        h = self.act5(h)
        h = self.norm5(h)

        h = torch.cat([h, z], dim=-1)
        h = self.fc3(h)
        h = self.act3(h)
        h = self.norm3(h)
        '''

        out = self.out_layer(h)
        return out


class ContextualHeteroGraphVAE(nn.Module):
    def __init__(self, ingre_fv_dim, direc_fv_dim, hidden_dim, latent_dim, out_dim, metadata, heads=1, dropout=0.0):
        super().__init__()
        self.encoder = HeteroEncoderHAN(
            ingre_fv_dim=ingre_fv_dim,
            direc_fv_dim=direc_fv_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            metadata=metadata,
            heads=heads,
            dropout=dropout
        )
        self.decoder = HeteroDecoder(
            latent_dim=latent_dim,
            context_dim=hidden_dim,  # context is from ingredient neighborhood
            hidden_dim=hidden_dim,
            out_dim=out_dim  # ingredient feature dim
        )

    def reparameterize(self, mu, logvar):
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        '''
        num_samples = 5
        std = torch.exp(0.5 * logvar)  # [B, D]
        B, D = mu.size()

        eps = torch.randn((num_samples, B, D), device=mu.device)  # [K, B, D]
        z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)       # [K, B, D]

        z_mean = z_samples.mean(dim=0)  # [B, D] ← 평균 over K
        return z_mean

    def forward(self, data):
        mu, logvar, ingre_context = self.encoder(data)
        z = self.reparameterize(mu, logvar)  # shape: [k, latent_dim]

        #z = torch.randn_like(mu) # → 여기서도 복원이 잘 되면, z는 무시당하고 있는 것

        if ingre_context.size(0) == 1 and z.size(0) > 1:
            # broadcast context for multiple masked nodes
            ingre_context = ingre_context.expand(z.size(0), -1)

        decoder_input = torch.cat([z, ingre_context], dim=-1)  # [k, latent + context]
        recon = self.decoder(decoder_input)  # [k, out_dim]

        #print("recon stats:", recon.min(), recon.max(), torch.isnan(recon).sum())
        return recon, mu, logvar


