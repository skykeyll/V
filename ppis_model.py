from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MeanAggregation, SumAggregation

try:
    import e3nn
    from e3nn import o3 as e3nn_o3
except Exception:
    e3nn = None
    e3nn_o3 = None


class CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(1).fill_(scale_init))

    def forward(self, coors: torch.Tensor) -> torch.Tensor:
        norm = coors.norm(dim=-1, keepdim=True)
        return coors / norm.clamp(min=self.eps) * self.scale


class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float = 0.2):
        super().__init__(aggr=None)
        self.edge_attr_dim = int(edge_attr_dim)
        self.node_aggr = SumAggregation()
        self.coord_aggr = MeanAggregation()
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_attr_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        pos_layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(pos_layer.weight, gain=0.001)
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            pos_layer,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.coors_norm = CoorsNorm()
        self.att_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        x,
        pos,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        residual = x if torch.is_tensor(x) else x[1]
        target = residual
        pos_target = pos if torch.is_tensor(pos) else pos[1]
        if edge_index.numel() == 0:
            return residual, pos_target
        if edge_attr is None:
            edge_attr = target.new_zeros((edge_index.size(1), self.edge_attr_dim))
        x_out, pos_out = self.propagate(
            edge_index=edge_index,
            x=x,
            pos=pos,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )
        return x_out + residual, pos_out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        pos_dir = pos_i - pos_j
        dist = torch.norm(pos_dir, dim=-1, keepdim=True)
        msg_in = torch.cat([self.node_norm(x_i), self.node_norm(x_j), dist, edge_attr], dim=-1)
        node_message = self.message_net(msg_in)
        node_message = self.att_mlp(node_message) * node_message
        pos_message = self.coors_norm(pos_dir) * self.pos_net(node_message)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1, 1)
            node_message = node_message * edge_weight
            pos_message = pos_message * edge_weight
        return node_message, pos_message

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        node_message, pos_message = inputs
        return self.node_aggr(node_message, index, ptr, dim_size), self.coord_aggr(pos_message, index, ptr, dim_size)

    def update(self, inputs, x, pos):
        node_message, pos_message = inputs
        x_target = x if torch.is_tensor(x) else x[1]
        pos_target = pos if torch.is_tensor(pos) else pos[1]
        x_new = self.update_net(torch.cat([x_target, node_message], dim=-1))
        pos_new = pos_target + pos_message
        return x_new, pos_new


class VDGStyleEGNNLayer(nn.Module):
    """VDG-style EGNN layer used for the Euclidean four-step pipeline.

    This layer expects `edge_index` in sender-first format: `[source, target]`.
    `edge_attr` is accepted for interface compatibility but intentionally ignored
    to match the EGNN implementation used by the VDG-PPIS model.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        dropout: float = 0.0,
        *,
        attention: bool = False,
        norm_coords: bool = True,
        norm_feats: bool = True,
        norm_coors_scale_init: float = 1e-2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.attention = bool(attention)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        else:
            self.att_mlp = None
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_norm = nn.LayerNorm(self.hidden_dim) if norm_feats else nn.Identity()
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coords else nn.Identity()

    def forward(
        self,
        x,
        pos,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        x_source, x_target = (x, x) if torch.is_tensor(x) else x
        pos_source, pos_target = (pos, pos) if torch.is_tensor(pos) else pos
        if edge_index is None or edge_index.numel() == 0:
            return x_target, pos_target

        src, dst = edge_index
        rel_pos = pos_target[dst] - pos_source[src]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_input = torch.cat(
            [self.node_norm(x_target[dst]), self.node_norm(x_source[src]), dist],
            dim=-1,
        )
        edge_message = self.edge_mlp(edge_input)
        if self.att_mlp is not None:
            edge_message = self.att_mlp(edge_message) * edge_message
        coord_scale = self.coord_mlp(edge_message)
        coord_message_per_edge = self.coors_norm(rel_pos) * coord_scale

        if edge_weight is not None:
            weight = edge_weight.view(-1, 1)
            edge_message = edge_message * weight
            coord_message_per_edge = coord_message_per_edge * weight

        node_message = _scatter_sum(edge_message, dst, x_target.size(0))
        coord_message = _scatter_mean(coord_message_per_edge, dst, pos_target.size(0))
        x_next = x_target + self.node_mlp(torch.cat([x_target, node_message], dim=-1))
        pos_next = pos_target + coord_message
        return x_next, pos_next


class EGNNEncoder(nn.Module):
    """Local geometry encoder. IPR context is intentionally kept outside coordinate updates."""

    def __init__(self, input_dim: int, hidden_dim: int, edge_attr_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [VDGStyleEGNNLayer(hidden_dim, edge_attr_dim=edge_attr_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        x = self.input_proj(x)
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr)
        return self.norm(x), pos


class IPRScorerLayer(MessagePassing):
    """Small feature-only scorer GNN over the original residue graph."""

    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float = 0.2):
        super().__init__(aggr="mean")
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        residual = x
        message = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        return residual + self.update_net(torch.cat([x, message], dim=-1))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.message_net(torch.cat([x_i, x_j, edge_attr], dim=-1))


class IPRScorerGNN(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int, num_centroids: int, context_layers: int, dropout: float):
        super().__init__()
        self.context_layers = nn.ModuleList(
            [IPRScorerLayer(hidden_dim, edge_attr_dim=edge_attr_dim, dropout=dropout) for _ in range(context_layers)]
        )
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_centroids),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for layer in self.context_layers:
            x = layer(x, edge_index, edge_attr)
        return self.score_mlp(x)


@dataclass
class IPRSharedState:
    assignment: torch.Tensor
    assignment_weight: torch.Tensor
    residue_to_virtual_edge_index: torch.Tensor
    virtual_to_residue_edge_index: torch.Tensor
    virtual_to_virtual_edge_index: torch.Tensor
    virtual_x: torch.Tensor
    virtual_pos: torch.Tensor


class IPRVirtualNodeLayer(nn.Module):
    """Hierarchical EGNN block: residue -> virtual, virtual -> virtual, virtual -> residue."""

    def __init__(self, hidden_dim: int, edge_attr_dim: int, dropout: float):
        super().__init__()
        self.residue_to_virtual = VDGStyleEGNNLayer(
            hidden_dim,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            attention=False,
            norm_coords=True,
            norm_feats=True,
        )
        self.virtual_to_virtual = VDGStyleEGNNLayer(
            hidden_dim,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            attention=False,
            norm_coords=True,
            norm_feats=True,
        )
        self.virtual_to_residue = VDGStyleEGNNLayer(
            hidden_dim,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            attention=False,
            norm_coords=True,
            norm_feats=True,
        )

    def forward(
        self,
        x_residue: torch.Tensor,
        pos_residue: torch.Tensor,
        x_virtual: torch.Tensor,
        pos_virtual: torch.Tensor,
        residue_to_virtual_edge_index: torch.Tensor,
        virtual_to_residue_edge_index: torch.Tensor,
        virtual_to_virtual_edge_index: torch.Tensor,
        assignment_weight: torch.Tensor,
    ):
        x_virtual, pos_virtual = self.residue_to_virtual(
            x=(x_residue, x_virtual),
            pos=(pos_residue, pos_virtual),
            edge_index=residue_to_virtual_edge_index,
            edge_weight=assignment_weight,
        )
        x_virtual, pos_virtual = self.virtual_to_virtual(
            x_virtual,
            pos_virtual,
            virtual_to_virtual_edge_index,
        )
        x_residue, pos_residue = self.virtual_to_residue(
            x=(x_virtual, x_residue),
            pos=(pos_virtual, pos_residue),
            edge_index=virtual_to_residue_edge_index,
            edge_weight=assignment_weight,
        )
        return x_residue, pos_residue, x_virtual, pos_virtual


class IPRVirtualNodeContext(nn.Module):
    """Build one deterministic balanced soft assignment for the shared virtual nodes."""

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_centroids: int,
        temperature: float,
        sinkhorn_iterations: int,
        eval_hard: bool,
        scorer_context_layers: int,
        aux_balance_weight: float,
        aux_entropy_weight: float,
        aux_compact_weight: float,
        aux_smooth_weight: float,
        aux_score_l2_weight: float,
        dropout: float,
    ):
        super().__init__()
        if num_centroids < 1:
            raise ValueError("num_centroids must be at least 1.")

        self.num_centroids = int(num_centroids)
        self.temperature = float(temperature)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.eval_hard = bool(eval_hard)
        self.aux_balance_weight = float(aux_balance_weight)
        self.aux_entropy_weight = float(aux_entropy_weight)
        self.aux_compact_weight = float(aux_compact_weight)
        self.aux_smooth_weight = float(aux_smooth_weight)
        self.aux_score_l2_weight = float(aux_score_l2_weight)

        self.scorer = IPRScorerGNN(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_centroids=self.num_centroids,
            context_layers=scorer_context_layers,
            dropout=0.0,
        )
        self.centroid_embedding = nn.Parameter(torch.empty(self.num_centroids, hidden_dim))
        nn.init.xavier_uniform_(self.centroid_embedding)

    def _balanced_sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        logits = scores / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        assignment = torch.exp(logits).clamp_min(eps)
        target_col_sum = float(scores.size(0)) / float(self.num_centroids)

        for _ in range(max(self.sinkhorn_iterations, 1)):
            assignment = assignment / assignment.sum(dim=1, keepdim=True).clamp_min(eps)
            assignment = assignment * (
                target_col_sum / assignment.sum(dim=0, keepdim=True).clamp_min(eps)
            )

        return assignment / assignment.sum(dim=1, keepdim=True).clamp_min(eps)

    def _hard_assignment(self, assignment: torch.Tensor) -> torch.Tensor:
        hard_idx = assignment.argmax(dim=-1, keepdim=True)
        hard_assignment = torch.zeros_like(assignment)
        hard_assignment.scatter_(1, hard_idx, 1.0)
        return hard_assignment

    def _assignment_entropy(self, assignment: torch.Tensor) -> torch.Tensor:
        if self.num_centroids == 1:
            return assignment.new_zeros(())
        entropy = -(assignment * torch.log(assignment.clamp_min(1e-12))).sum(dim=-1).mean()
        return entropy / torch.log(assignment.new_tensor(float(self.num_centroids)))

    def _balance_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        usage = assignment.sum(dim=0)
        usage_probs = usage / usage.sum().clamp_min(1e-12)
        uniform = usage_probs.new_full(usage_probs.shape, 1.0 / float(self.num_centroids))
        return (usage_probs * (torch.log(usage_probs.clamp_min(1e-12)) - torch.log(uniform))).sum()

    def _compact_loss(self, assignment: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        usage = assignment.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
        virtual_pos = assignment.t().matmul(pos) / usage
        dist2 = (pos.unsqueeze(1) - virtual_pos.unsqueeze(0)).pow(2).sum(dim=-1)
        pos_scale2 = (pos - pos.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean().clamp_min(1.0)
        return (assignment * dist2).sum() / assignment.sum().clamp_min(1e-12) / pos_scale2

    def _smooth_loss(self, assignment: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return assignment.new_zeros(())
        src, dst = edge_index
        keep = src != dst
        if not bool(keep.any()):
            return assignment.new_zeros(())
        diff = assignment[src[keep]] - assignment[dst[keep]]
        return diff.pow(2).sum(dim=-1).mean()

    def _compute_aux_loss(
        self,
        scores: torch.Tensor,
        assignment: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return scores.new_zeros(())

        balance_loss = self._balance_loss(assignment)
        entropy_loss = self._assignment_entropy(assignment)
        compact_loss = self._compact_loss(assignment, pos)
        smooth_loss = self._smooth_loss(assignment, edge_index)
        score_l2_loss = scores.pow(2).mean()

        return (
            self.aux_balance_weight * balance_loss
            + self.aux_entropy_weight * entropy_loss
            + self.aux_compact_weight * compact_loss
            + self.aux_smooth_weight * smooth_loss
            + self.aux_score_l2_weight * score_l2_loss
        )

    def _diagnostics(
        self,
        scores: torch.Tensor,
        assignment: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            hard_assignment = self._hard_assignment(assignment)
            hard_usage = hard_assignment.sum(dim=0)
            soft_usage = assignment.detach().sum(dim=0)
            entropy = self._assignment_entropy(assignment)
            compact = self._compact_loss(assignment, pos)
            smooth = self._smooth_loss(assignment, edge_index)
            balance = self._balance_loss(assignment)
            row_error = (assignment.sum(dim=1) - 1.0).abs().max()
            target_usage = float(assignment.size(0)) / float(self.num_centroids)
            balance_error = (soft_usage - target_usage).abs().max()
            return {
                "hard_empty": int((hard_usage <= 0).sum().item()),
                "hard_usage_min": float(hard_usage.min().item()),
                "hard_usage_max": float(hard_usage.max().item()),
                "soft_usage_min": float(soft_usage.min().item()),
                "soft_usage_max": float(soft_usage.max().item()),
                "sinkhorn_row_error": float(row_error.item()),
                "sinkhorn_balance_error": float(balance_error.item()),
                "score_entropy": float(entropy.item()),
                "assignment_entropy": float(entropy.item()),
                "compact_loss": float(compact.item()),
                "smooth_loss": float(smooth.item()),
                "balance_loss": float(balance.item()),
            }

    def _build_virtual_edges(self, num_residues: int, device: torch.device):
        residue_index = torch.arange(num_residues, device=device).repeat_interleave(self.num_centroids)
        virtual_index = torch.arange(self.num_centroids, device=device).repeat(num_residues)
        residue_to_virtual_edge_index = torch.stack([residue_index, virtual_index], dim=0)
        virtual_to_residue_edge_index = torch.stack([virtual_index, residue_index], dim=0)

        if self.num_centroids == 1:
            virtual_to_virtual_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            src = torch.arange(self.num_centroids, device=device).repeat_interleave(self.num_centroids)
            dst = torch.arange(self.num_centroids, device=device).repeat(self.num_centroids)
            keep = src != dst
            virtual_to_virtual_edge_index = torch.stack([src[keep], dst[keep]], dim=0)

        return residue_to_virtual_edge_index, virtual_to_residue_edge_index, virtual_to_virtual_edge_index

    def _initialize_virtual_nodes(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        assignment: torch.Tensor,
    ):
        denom = assignment.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
        virtual_x = assignment.t().matmul(x) / denom
        virtual_pos = assignment.t().matmul(pos) / denom
        return virtual_x + self.centroid_embedding, virtual_pos

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_aux: bool = False,
        force_soft_assignment: bool = False,
        return_diagnostics: bool = False,
        score_x: torch.Tensor = None,
    ):
        scorer_input = x if score_x is None else score_x
        scores = self.scorer(scorer_input, edge_index, edge_attr)
        assignment = self._balanced_sinkhorn(scores)
        if self.eval_hard and not self.training and not force_soft_assignment:
            assignment = self._hard_assignment(assignment)
        aux_loss = self._compute_aux_loss(scores, assignment, pos, edge_index) if return_aux else scores.new_zeros(())

        virtual_x, virtual_pos = self._initialize_virtual_nodes(x, pos, assignment)
        residue_to_virtual_edge_index, virtual_to_residue_edge_index, virtual_to_virtual_edge_index = (
            self._build_virtual_edges(x.size(0), x.device)
        )
        assignment_weight = assignment.reshape(-1)
        if self.training:
            active_edge_mask = torch.ones_like(assignment_weight, dtype=torch.bool)
        else:
            active_edge_mask = assignment_weight.detach() > 0.0
            if not bool(active_edge_mask.any()):
                active_edge_mask = torch.ones_like(assignment_weight, dtype=torch.bool)
        residue_to_virtual_edge_index = residue_to_virtual_edge_index[:, active_edge_mask]
        virtual_to_residue_edge_index = virtual_to_residue_edge_index[:, active_edge_mask]
        assignment_weight = assignment_weight[active_edge_mask]
        state = IPRSharedState(
            assignment=assignment,
            assignment_weight=assignment_weight,
            residue_to_virtual_edge_index=residue_to_virtual_edge_index,
            virtual_to_residue_edge_index=virtual_to_residue_edge_index,
            virtual_to_virtual_edge_index=virtual_to_virtual_edge_index,
            virtual_x=virtual_x,
            virtual_pos=virtual_pos,
        )

        if return_aux and return_diagnostics:
            return state, aux_loss, self._diagnostics(scores, assignment, pos, edge_index)
        if return_aux:
            return state, aux_loss
        if return_diagnostics:
            return state, self._diagnostics(scores, assignment, pos, edge_index)
        return state


class IPRVirtualNodeRefiner(nn.Module):
    """Reuse one fixed H while updating residue and virtual states downstream."""

    def __init__(self, hidden_dim: int, edge_attr_dim: int, message_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [IPRVirtualNodeLayer(hidden_dim, edge_attr_dim=edge_attr_dim, dropout=dropout) for _ in range(message_layers)]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor, state: IPRSharedState):
        virtual_x = state.virtual_x
        virtual_pos = state.virtual_pos
        for layer in self.layers:
            x, pos, virtual_x, virtual_pos = layer(
                x_residue=x,
                pos_residue=pos,
                x_virtual=virtual_x,
                pos_virtual=virtual_pos,
                residue_to_virtual_edge_index=state.residue_to_virtual_edge_index,
                virtual_to_residue_edge_index=state.virtual_to_residue_edge_index,
                virtual_to_virtual_edge_index=state.virtual_to_virtual_edge_index,
                assignment_weight=state.assignment_weight,
            )

        next_state = IPRSharedState(
            assignment=state.assignment,
            assignment_weight=state.assignment_weight,
            residue_to_virtual_edge_index=state.residue_to_virtual_edge_index,
            virtual_to_residue_edge_index=state.virtual_to_residue_edge_index,
            virtual_to_virtual_edge_index=state.virtual_to_virtual_edge_index,
            virtual_x=virtual_x,
            virtual_pos=virtual_pos,
        )
        return x, pos, next_state


class EGNNIPRBlock(nn.Module):
    """One residue EGNN layer followed by a refinement on the shared expanded graph."""

    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        dropout: float,
        use_ipr_virtual_nodes: bool,
        ipr_virtual_message_layers: int,
    ):
        super().__init__()
        self.egnn = VDGStyleEGNNLayer(
            hidden_dim,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            attention=True,
            norm_coords=False,
            norm_feats=False,
        )
        self.ipr_context = (
            IPRVirtualNodeRefiner(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                message_layers=ipr_virtual_message_layers,
                dropout=dropout,
            )
            if use_ipr_virtual_nodes
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        state: IPRSharedState = None,
    ):
        local_x, local_pos = self.egnn(x, pos, edge_index, edge_attr)
        if self.ipr_context is not None:
            if state is None:
                raise ValueError("IPRSharedState is required when use_ipr_virtual_nodes=True.")
            x, pos, state = self.ipr_context(local_x, local_pos, state)
        else:
            x, pos = local_x, local_pos
        return x, pos, state


class EGNNHierarchicalEncoder(nn.Module):
    """Sample one shared H, then reuse it across all downstream EGNN/IPR layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_attr_dim: int,
        num_layers: int,
        dropout: float,
        use_ipr_virtual_nodes: bool,
        ipr_num_virtual_nodes: int,
        ipr_temperature: float,
        ipr_sinkhorn_iterations: int,
        ipr_eval_hard: bool,
        ipr_scorer_context_layers: int,
        ipr_virtual_message_layers: int,
        ipr_aux_entropy_weight: float,
        ipr_aux_balance_weight: float,
        ipr_aux_compact_weight: float,
        ipr_aux_smooth_weight: float,
        ipr_aux_score_l2_weight: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ipr_score_proj = nn.Linear(input_dim, hidden_dim) if use_ipr_virtual_nodes else None
        self.ipr_sampler = (
            IPRVirtualNodeContext(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                num_centroids=ipr_num_virtual_nodes,
                temperature=ipr_temperature,
                sinkhorn_iterations=ipr_sinkhorn_iterations,
                eval_hard=ipr_eval_hard,
                scorer_context_layers=ipr_scorer_context_layers,
                aux_balance_weight=ipr_aux_balance_weight,
                aux_entropy_weight=ipr_aux_entropy_weight,
                aux_compact_weight=ipr_aux_compact_weight,
                aux_smooth_weight=ipr_aux_smooth_weight,
                aux_score_l2_weight=ipr_aux_score_l2_weight,
                dropout=dropout,
            )
            if use_ipr_virtual_nodes
            else None
        )
        self.blocks = nn.ModuleList(
            [
                EGNNIPRBlock(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    dropout=dropout,
                    use_ipr_virtual_nodes=use_ipr_virtual_nodes,
                    ipr_virtual_message_layers=ipr_virtual_message_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_aux: bool = False,
    ):
        raw_x = x
        x = self.input_proj(raw_x)
        aux_loss = x.new_zeros(())
        state = None
        if self.ipr_sampler is not None:
            score_x = self.ipr_score_proj(raw_x)
            if return_aux:
                state, aux_loss = self.ipr_sampler(x, pos, edge_index, edge_attr, return_aux=True, score_x=score_x)
            else:
                state = self.ipr_sampler(x, pos, edge_index, edge_attr, score_x=score_x)
        for block in self.blocks:
            x, pos, state = block(x, pos, edge_index, edge_attr, state)
        return self.norm(x), pos, aux_loss

    def ipr_warmup_loss(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        if self.ipr_sampler is None:
            projected = self.input_proj(x)
            return projected.new_zeros(()), {}
        projected = self.input_proj(x)
        score_x = self.ipr_score_proj(x)
        _, aux_loss, diagnostics = self.ipr_sampler(
            projected,
            pos,
            edge_index,
            edge_attr,
            return_aux=True,
            force_soft_assignment=True,
            return_diagnostics=True,
            score_x=score_x,
        )
        return aux_loss, diagnostics


def _require_e3nn() -> None:
    if e3nn is None or e3nn_o3 is None:
        raise ImportError("ResidueDualEqui requires e3nn. Install e3nn or use encoder_type='egnn'.")


def _empty_edge_index(device: torch.device) -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _to_receiver_first(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        device = edge_index.device if edge_index is not None else torch.device("cpu")
        return _empty_edge_index(device)
    return torch.stack([edge_index[1], edge_index[0]], dim=0)


def _remove_self_loops(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if edge_index is None or edge_index.numel() == 0:
        device = edge_index.device if edge_index is not None else torch.device("cpu")
        return _empty_edge_index(device), edge_attr

    keep = edge_index[0] != edge_index[1]
    if bool(keep.all()):
        return edge_index, edge_attr

    edge_index = edge_index[:, keep]
    if edge_attr is not None and edge_attr.size(0) == keep.size(0):
        edge_attr = edge_attr[keep]
    return edge_index, edge_attr


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    output = src.new_zeros((dim_size, *src.shape[1:]))
    if src.numel() > 0:
        output.index_add_(0, index, src)
    return output


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    output = _scatter_sum(src, index, dim_size)
    count = src.new_zeros((dim_size, *([1] * (src.dim() - 1))))
    if index.numel() > 0:
        ones = torch.ones((index.numel(), *([1] * (src.dim() - 1))), dtype=src.dtype, device=src.device)
        count.index_add_(0, index, ones)
    return output / count.clamp_min(1.0)


def _radius_graph_single(
    pos: torch.Tensor,
    cutoff: float,
    max_neighbors: int = None,
) -> torch.Tensor:
    num_nodes = pos.size(0)
    if num_nodes <= 1:
        return _empty_edge_index(pos.device)

    distances = torch.cdist(pos, pos)
    distances.fill_diagonal_(float("inf"))

    if max_neighbors is not None:
        k = min(max(int(max_neighbors), 0), num_nodes - 1)
        if k == 0:
            return _empty_edge_index(pos.device)
        neighbor_distances, neighbors = distances.topk(k, largest=False, dim=1)
        row = torch.arange(num_nodes, device=pos.device).repeat_interleave(k)
        col = neighbors.reshape(-1)
        keep = neighbor_distances.reshape(-1) <= float(cutoff)
        return torch.stack([row[keep], col[keep]], dim=0)

    mask = distances <= float(cutoff)
    return mask.nonzero(as_tuple=False).t().contiguous()


class ResidueBaseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation=nn.SiLU(),
        last_act: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.Dropout(dropout),
            activation,
            nn.Linear(hidden_dim, output_dim, bias=bias),
            activation if last_act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResidueSHInit(nn.Module):
    def __init__(self, hidden_dim: int, lmax: int, dropout: float):
        super().__init__()
        _require_e3nn()
        self.sh_irreps = e3nn_o3.Irreps.spherical_harmonics(lmax)
        self.spherical_harmonics = e3nn_o3.SphericalHarmonics(
            self.sh_irreps,
            normalize=True,
            normalization="norm",
        )
        self.sh_coeff = e3nn_o3.FullyConnectedTensorProduct(
            self.sh_irreps,
            "1x0e",
            self.sh_irreps,
            shared_weights=False,
        )
        self.sh_mlp = ResidueBaseMLP(
            input_dim=2 * hidden_dim + 1,
            hidden_dim=hidden_dim,
            output_dim=self.sh_coeff.weight_numel,
            dropout=dropout,
            last_act=True,
        )

    @property
    def output_dim(self) -> int:
        return self.sh_irreps.dim

    def forward(self, residue_h: torch.Tensor, residue_pos: torch.Tensor, residue_edge_index: torch.Tensor) -> torch.Tensor:
        if residue_edge_index is None or residue_edge_index.numel() == 0:
            return residue_h.new_zeros((residue_h.size(0), self.output_dim))

        row, col = residue_edge_index
        rel_pos = residue_pos[row] - residue_pos[col]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        coeff = self.sh_mlp(torch.cat([dist, residue_h[row], residue_h[col]], dim=-1))
        rel_sh = self.spherical_harmonics(rel_pos).detach()
        scalar_one = torch.ones((rel_sh.size(0), 1), dtype=rel_sh.dtype, device=rel_sh.device)
        rel_sh = self.sh_coeff(rel_sh, scalar_one, coeff)
        return _scatter_mean(rel_sh, row, residue_h.size(0))


class ResidueInnerProductSH(nn.Module):
    def __init__(self, sh_irreps):
        super().__init__()
        self.sh_irreps = sh_irreps

    def forward(self, edge_index: torch.Tensor, residue_sh: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        temp = residue_sh[row] * residue_sh[col]
        in_prod = residue_sh.new_zeros((edge_index.size(1), self.sh_irreps.lmax + 1))

        start = 0
        for _, ir in self.sh_irreps:
            in_prod[:, ir.l] = temp[:, start:start + ir.dim].sum(dim=-1)
            start += ir.dim

        return in_prod / (torch.norm(in_prod, dim=1, keepdim=True).detach() + 1.0)


class ResidueSphericalHarmonicGraph(nn.Module):
    def __init__(self, threshold: float, max_neighbors: int = None):
        super().__init__()
        self.threshold = float(threshold)
        self.max_neighbors = max_neighbors

    def forward(self, residue_sh: torch.Tensor) -> torch.Tensor:
        num_nodes = residue_sh.size(0)
        if num_nodes <= 1:
            return _empty_edge_index(residue_sh.device)

        residue_sh_norm = F.normalize(residue_sh, p=2, dim=-1)
        similarity = residue_sh_norm @ residue_sh_norm.t()
        similarity.fill_diagonal_(-float("inf"))

        if self.max_neighbors is not None:
            k = min(max(int(self.max_neighbors), 0), num_nodes - 1)
            if k == 0:
                return _empty_edge_index(residue_sh.device)
            values, neighbors = similarity.topk(k, largest=True, dim=1)
            row = torch.arange(num_nodes, device=residue_sh.device).repeat_interleave(k)
            col = neighbors.reshape(-1)
            keep = values.reshape(-1) > self.threshold
            return torch.stack([row[keep], col[keep]], dim=0)

        mask = similarity > self.threshold
        return mask.nonzero(as_tuple=False).t().contiguous()


class ResidueEuclideanSpaceBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_heads: int,
        lmax: int,
        inner_product: ResidueInnerProductSH = None,
        dropout: float = 0.0,
        use_cross_space: bool = True,
        use_cross_space_gate: bool = False,
        use_cross_space_semantic: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by residue_num_heads.")

        self.hidden_dim = hidden_dim
        self.edge_attr_dim = int(edge_attr_dim)
        self.num_heads = int(num_heads)
        self.inner_product = inner_product
        self.use_cross_space = bool(use_cross_space)
        self.use_cross_space_gate = bool(use_cross_space_gate and self.use_cross_space)
        self.use_cross_space_semantic = bool(use_cross_space_semantic and self.use_cross_space)
        self.pos_scale_factor = nn.Parameter(torch.tensor([0.1]))

        self.h_mlp = ResidueBaseMLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.msg_mlp = ResidueBaseMLP(
            input_dim=2 * hidden_dim + 1 + self.edge_attr_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * self.num_heads + self.num_heads + 1,
            dropout=dropout,
        )
        self.cross_mlp = (
            ResidueBaseMLP(
                input_dim=lmax + 2 + (2 * hidden_dim if self.use_cross_space_semantic else 0),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim * self.num_heads + self.num_heads + 1,
                dropout=dropout,
            )
            if self.use_cross_space
            else None
        )
        self.cross_gate = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if self.use_cross_space_gate
            else None
        )
        if self.cross_gate is not None:
            nn.init.constant_(self.cross_gate[-2].bias, -2.0)

    def _edge_attr_or_zeros(self, residue_h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.edge_attr_dim == 0:
            return residue_h.new_zeros((edge_index.size(1), 0))
        if edge_attr is None or edge_attr.size(0) != edge_index.size(1):
            return residue_h.new_zeros((edge_index.size(1), self.edge_attr_dim))
        return edge_attr

    def forward(
        self,
        residue_h: torch.Tensor,
        residue_pos: torch.Tensor,
        residue_edge_index_eu: torch.Tensor,
        residue_edge_attr_eu: torch.Tensor,
        residue_edge_index_sh: torch.Tensor,
        residue_sh: torch.Tensor,
        residue_edge_weight_eu: torch.Tensor = None,
        residue_edge_weight_sh: torch.Tensor = None,
    ):
        num_nodes = residue_h.size(0)
        h_update_eu = residue_h.new_zeros((num_nodes, self.num_heads, self.hidden_dim))
        pos_update_eu = residue_pos.new_zeros((num_nodes, 3))
        has_eu_edges = residue_edge_index_eu is not None and residue_edge_index_eu.numel() > 0

        if has_eu_edges:
            row, col = residue_edge_index_eu
            rel_pos_eu = residue_pos[row] - residue_pos[col]
            dist_eu = torch.norm(rel_pos_eu, dim=-1, keepdim=True)
            edge_attr = self._edge_attr_or_zeros(residue_h, residue_edge_index_eu, residue_edge_attr_eu)
            msg_inputs = torch.cat([residue_h[row], residue_h[col], dist_eu, edge_attr], dim=-1)
            msg_outputs = self.msg_mlp(msg_inputs)
            msg, attention_logits, pos_scale = torch.split(
                msg_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1,
            )
            attention = torch.sigmoid(attention_logits)
            if residue_edge_weight_eu is not None:
                attention = attention * residue_edge_weight_eu.view(-1, 1)
            msg = msg.view(-1, self.num_heads, self.hidden_dim)
            h_update_eu = _scatter_sum(msg * attention.unsqueeze(-1), row, num_nodes)
            pos_update_eu = _scatter_sum(
                rel_pos_eu * torch.tanh(pos_scale) * attention.mean(dim=1, keepdim=True),
                row,
                num_nodes,
            )

        h_update_cross = residue_h.new_zeros((num_nodes, self.num_heads, self.hidden_dim))
        pos_update_cross = residue_pos.new_zeros((num_nodes, 3))
        has_sh_edges = residue_edge_index_sh is not None and residue_edge_index_sh.numel() > 0

        if self.use_cross_space and has_sh_edges:
            row, col = residue_edge_index_sh
            rel_pos_sh = residue_pos[row] - residue_pos[col]
            dist_sh = torch.norm(rel_pos_sh, dim=-1, keepdim=True)
            in_prod = self.inner_product(residue_edge_index_sh, residue_sh)
            cross_inputs = [in_prod, dist_sh]
            if self.use_cross_space_semantic:
                cross_inputs = [residue_h[row], residue_h[col], *cross_inputs]
            cross_outputs = self.cross_mlp(torch.cat(cross_inputs, dim=-1))
            cross_msg, cross_attention_logits, pos_scale_cross = torch.split(
                cross_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1,
            )
            cross_attention = torch.sigmoid(cross_attention_logits)
            if residue_edge_weight_sh is not None:
                cross_attention = cross_attention * residue_edge_weight_sh.view(-1, 1)
            cross_msg = cross_msg.view(-1, self.num_heads, self.hidden_dim)
            h_update_cross = _scatter_sum(cross_msg * cross_attention.unsqueeze(-1), row, num_nodes)
            pos_update_cross = _scatter_sum(
                rel_pos_sh * torch.tanh(pos_scale_cross) * cross_attention.mean(dim=1, keepdim=True),
                row,
                num_nodes,
            )

        if not has_eu_edges and not (self.use_cross_space and has_sh_edges):
            return torch.zeros_like(residue_h), torch.zeros_like(residue_pos)

        h_update_eu = h_update_eu.mean(dim=1)
        h_update_cross = h_update_cross.mean(dim=1)
        if self.cross_gate is not None:
            cross_gate = self.cross_gate(torch.cat([residue_h, h_update_cross], dim=-1))
            h_update_cross = h_update_cross * cross_gate
            pos_update_cross = pos_update_cross * cross_gate.mean(dim=-1, keepdim=True)
        h_update = h_update_eu + h_update_cross
        h_update = self.h_mlp(torch.cat([residue_h, h_update], dim=-1))
        pos_update = (pos_update_eu + pos_update_cross) * self.pos_scale_factor
        return h_update, pos_update


class ResidueSHSpaceBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        lmax: int,
        inner_product: ResidueInnerProductSH,
        spherical_harmonics,
        dropout: float = 0.0,
        use_cross_space: bool = True,
        use_cross_space_gate: bool = False,
        use_cross_space_semantic: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by residue_num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = int(num_heads)
        self.inner_product = inner_product
        self.spherical_harmonics = spherical_harmonics
        self.use_cross_space = bool(use_cross_space)
        self.use_cross_space_gate = bool(use_cross_space_gate and self.use_cross_space)
        self.use_cross_space_semantic = bool(use_cross_space_semantic and self.use_cross_space)
        self.residue_sh_scale_factor = nn.Parameter(torch.tensor([0.1]))

        self.h_mlp = ResidueBaseMLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.msg_mlp = ResidueBaseMLP(
            input_dim=2 * hidden_dim + (lmax + 1),
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * self.num_heads + self.num_heads + 1,
            dropout=dropout,
        )
        self.cross_mlp = (
            ResidueBaseMLP(
                input_dim=lmax + 2 + (2 * hidden_dim if self.use_cross_space_semantic else 0),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim * self.num_heads + self.num_heads + 1,
                dropout=dropout,
            )
            if self.use_cross_space
            else None
        )
        self.cross_gate = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            if self.use_cross_space_gate
            else None
        )
        if self.cross_gate is not None:
            nn.init.constant_(self.cross_gate[-2].bias, -2.0)

    def forward(
        self,
        residue_h: torch.Tensor,
        residue_pos: torch.Tensor,
        residue_edge_index_eu: torch.Tensor,
        residue_edge_index_sh: torch.Tensor,
        residue_sh: torch.Tensor,
        residue_edge_weight_eu: torch.Tensor = None,
        residue_edge_weight_sh: torch.Tensor = None,
    ):
        num_nodes = residue_h.size(0)
        residue_sh_dim = residue_sh.size(-1)
        h_update_sh = residue_h.new_zeros((num_nodes, self.num_heads, self.hidden_dim))
        residue_sh_update = residue_sh.new_zeros((num_nodes, residue_sh_dim))
        has_sh_edges = residue_edge_index_sh is not None and residue_edge_index_sh.numel() > 0

        if has_sh_edges:
            row, col = residue_edge_index_sh
            in_prod_sh = self.inner_product(residue_edge_index_sh, residue_sh)
            msg_inputs = torch.cat([in_prod_sh, residue_h[row], residue_h[col]], dim=-1)
            msg_outputs = self.msg_mlp(msg_inputs)
            msg, attention_logits, residue_sh_scale = torch.split(
                msg_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1,
            )
            attention = torch.sigmoid(attention_logits)
            if residue_edge_weight_sh is not None:
                attention = attention * residue_edge_weight_sh.view(-1, 1)
            msg = msg.view(-1, self.num_heads, self.hidden_dim)
            h_update_sh = _scatter_sum(msg * attention.unsqueeze(-1), row, num_nodes)
            residue_sh_update = _scatter_sum(
                residue_sh[col] * torch.tanh(residue_sh_scale) * attention.mean(dim=1, keepdim=True),
                row,
                num_nodes,
            )

        h_update_cross = residue_h.new_zeros((num_nodes, self.num_heads, self.hidden_dim))
        residue_sh_update_cross = residue_sh.new_zeros((num_nodes, residue_sh_dim))
        has_eu_edges = residue_edge_index_eu is not None and residue_edge_index_eu.numel() > 0

        if self.use_cross_space and has_eu_edges:
            row, col = residue_edge_index_eu
            rel_pos_eu = residue_pos[row] - residue_pos[col]
            dist_eu = torch.norm(rel_pos_eu, dim=-1, keepdim=True)
            in_prod_eu = self.inner_product(residue_edge_index_eu, residue_sh)
            cross_inputs = [in_prod_eu, dist_eu]
            if self.use_cross_space_semantic:
                cross_inputs = [residue_h[row], residue_h[col], *cross_inputs]
            cross_outputs = self.cross_mlp(torch.cat(cross_inputs, dim=-1))
            cross_msg, cross_attention_logits, rel_sh_cross_scale = torch.split(
                cross_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1,
            )
            cross_attention = torch.sigmoid(cross_attention_logits)
            if residue_edge_weight_eu is not None:
                cross_attention = cross_attention * residue_edge_weight_eu.view(-1, 1)
            cross_msg = cross_msg.view(-1, self.num_heads, self.hidden_dim)
            h_update_cross = _scatter_sum(cross_msg * cross_attention.unsqueeze(-1), row, num_nodes)
            rel_sh = self.spherical_harmonics(rel_pos_eu).detach()
            residue_sh_update_cross = _scatter_sum(
                rel_sh * torch.tanh(rel_sh_cross_scale) * cross_attention.mean(dim=1, keepdim=True),
                row,
                num_nodes,
            )

        if not has_sh_edges and not (self.use_cross_space and has_eu_edges):
            return torch.zeros_like(residue_h), torch.zeros_like(residue_sh)

        h_update_sh = h_update_sh.mean(dim=1)
        h_update_cross = h_update_cross.mean(dim=1)
        if self.cross_gate is not None:
            cross_gate = self.cross_gate(torch.cat([residue_h, h_update_cross], dim=-1))
            h_update_cross = h_update_cross * cross_gate
            residue_sh_update_cross = residue_sh_update_cross * cross_gate.mean(dim=-1, keepdim=True)
        h_update = h_update_sh + h_update_cross
        h_update = self.h_mlp(torch.cat([residue_h, h_update], dim=-1))
        residue_sh_update = (residue_sh_update + residue_sh_update_cross) * self.residue_sh_scale_factor
        return h_update, residue_sh_update


class ResidueDualEquiLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_heads: int,
        lmax: int,
        dropout: float,
        use_sh_branch: bool = True,
        use_cross_space: bool = True,
        use_cross_space_gate: bool = False,
        use_cross_space_semantic: bool = False,
    ):
        super().__init__()
        self.use_sh_branch = bool(use_sh_branch)
        self.use_cross_space = bool(use_cross_space and use_sh_branch)
        self.use_cross_space_gate = bool(use_cross_space_gate and self.use_cross_space)
        self.use_cross_space_semantic = bool(use_cross_space_semantic and self.use_cross_space)
        self.uses_spherical_features = self.use_sh_branch or self.use_cross_space

        if self.uses_spherical_features:
            _require_e3nn()
            self.sh_irreps = e3nn_o3.Irreps.spherical_harmonics(lmax)
            self.inner_product = ResidueInnerProductSH(self.sh_irreps)
            self.spherical_harmonics = e3nn_o3.SphericalHarmonics(
                self.sh_irreps,
                normalize=True,
                normalization="norm",
            )
        else:
            self.sh_irreps = None
            self.inner_product = None
            self.spherical_harmonics = None

        self.eu_layer = ResidueEuclideanSpaceBlock(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_heads=num_heads,
            lmax=lmax,
            inner_product=self.inner_product,
            dropout=dropout,
            use_cross_space=self.use_cross_space,
            use_cross_space_gate=self.use_cross_space_gate,
            use_cross_space_semantic=self.use_cross_space_semantic,
        )
        self.sh_layer = (
            ResidueSHSpaceBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                lmax=lmax,
                inner_product=self.inner_product,
                spherical_harmonics=self.spherical_harmonics,
                dropout=dropout,
                use_cross_space=self.use_cross_space,
                use_cross_space_gate=self.use_cross_space_gate,
                use_cross_space_semantic=self.use_cross_space_semantic,
            )
            if self.use_sh_branch
            else None
        )

    def forward(
        self,
        residue_h: torch.Tensor,
        residue_pos: torch.Tensor,
        residue_edge_index_eu: torch.Tensor,
        residue_edge_attr_eu: torch.Tensor,
        residue_edge_index_sh: torch.Tensor,
        residue_sh: torch.Tensor,
        residue_edge_weight_eu: torch.Tensor = None,
        residue_edge_weight_sh: torch.Tensor = None,
    ):
        h_update_eu, pos_update = self.eu_layer(
            residue_h,
            residue_pos,
            residue_edge_index_eu,
            residue_edge_attr_eu,
            residue_edge_index_sh,
            residue_sh,
            residue_edge_weight_eu=residue_edge_weight_eu,
            residue_edge_weight_sh=residue_edge_weight_sh,
        )
        if self.sh_layer is None:
            h_update_sh = torch.zeros_like(residue_h)
            residue_sh_update = torch.zeros_like(residue_sh)
        else:
            h_update_sh, residue_sh_update = self.sh_layer(
                residue_h,
                residue_pos,
                residue_edge_index_eu,
                residue_edge_index_sh,
                residue_sh,
                residue_edge_weight_eu=residue_edge_weight_eu,
                residue_edge_weight_sh=residue_edge_weight_sh,
            )
        return residue_h + h_update_eu + h_update_sh, residue_pos + pos_update, residue_sh + residue_sh_update


class ResidueVirtualReturnDualSpace(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_heads: int,
        lmax: int,
        dropout: float,
        use_sh_branch: bool,
        use_cross_space: bool,
        use_cross_space_gate: bool,
        use_cross_space_semantic: bool,
        sh_cutoff: float,
        max_neighbors: int = None,
    ):
        super().__init__()
        self.edge_attr_dim = int(edge_attr_dim)
        self.sh_cutoff = float(sh_cutoff)
        self.max_neighbors = max_neighbors
        self.return_layer = ResidueDualEquiLayer(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_heads=num_heads,
            lmax=lmax,
            dropout=dropout,
            use_sh_branch=use_sh_branch,
            use_cross_space=use_cross_space,
            use_cross_space_gate=use_cross_space_gate,
            use_cross_space_semantic=use_cross_space_semantic,
        )

    def _pool_virtual_sh(self, residue_sh: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        denom = assignment.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
        return assignment.t().matmul(residue_sh) / denom

    def _build_return_sh_graph(self, residue_sh: torch.Tensor, virtual_sh: torch.Tensor) -> torch.Tensor:
        num_residues = residue_sh.size(0)
        num_virtual = virtual_sh.size(0)
        if num_residues == 0 or num_virtual == 0:
            return _empty_edge_index(residue_sh.device)

        residue_norm = F.normalize(residue_sh, p=2, dim=-1)
        virtual_norm = F.normalize(virtual_sh, p=2, dim=-1)
        similarity = residue_norm @ virtual_norm.t()

        if self.max_neighbors is not None:
            k = min(max(int(self.max_neighbors), 0), num_virtual)
            if k == 0:
                return _empty_edge_index(residue_sh.device)
            values, virtual_idx = similarity.topk(k, largest=True, dim=1)
            residue_idx = torch.arange(num_residues, device=residue_sh.device).repeat_interleave(k)
            virtual_idx = virtual_idx.reshape(-1)
            keep = values.reshape(-1) > self.sh_cutoff
            if not bool(keep.any()):
                return _empty_edge_index(residue_sh.device)
            return torch.stack([residue_idx[keep], virtual_idx[keep] + num_residues], dim=0)

        residue_idx, virtual_idx = (similarity > self.sh_cutoff).nonzero(as_tuple=True)
        if residue_idx.numel() == 0:
            return _empty_edge_index(residue_sh.device)
        return torch.stack([residue_idx, virtual_idx + num_residues], dim=0)

    def forward(
        self,
        residue_h: torch.Tensor,
        residue_pos: torch.Tensor,
        residue_sh: torch.Tensor,
        virtual_h: torch.Tensor,
        virtual_pos: torch.Tensor,
        state: IPRSharedState,
        virtual_sh: torch.Tensor = None,
    ):
        if (
            virtual_h is None
            or virtual_pos is None
        ):
            return residue_h, residue_pos, residue_sh, virtual_sh

        num_residues = residue_h.size(0)
        virtual_sh = self._pool_virtual_sh(residue_sh, state.assignment)
        if state.virtual_to_residue_edge_index.numel() == 0:
            return residue_h, residue_pos, residue_sh, virtual_sh

        combined_h = torch.cat([residue_h, virtual_h], dim=0)
        combined_pos = torch.cat([residue_pos, virtual_pos], dim=0)
        combined_sh = torch.cat([residue_sh, virtual_sh], dim=0)

        virtual_src, residue_dst = state.virtual_to_residue_edge_index
        return_edge_index = torch.stack([residue_dst, virtual_src + num_residues], dim=0)
        return_edge_attr = residue_h.new_zeros((return_edge_index.size(1), self.edge_attr_dim))
        return_sh_edge_index = self._build_return_sh_graph(residue_sh, virtual_sh)

        updated_h, updated_pos, updated_sh = self.return_layer(
            combined_h,
            combined_pos,
            return_edge_index,
            return_edge_attr,
            return_sh_edge_index,
            combined_sh,
            residue_edge_weight_eu=state.assignment_weight,
            residue_edge_weight_sh=None,
        )
        return updated_h[:num_residues], updated_pos[:num_residues], updated_sh[:num_residues], virtual_sh


class ResidueDualEquiIPRBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_attr_dim: int,
        num_heads: int,
        lmax: int,
        dropout: float,
        use_ipr_virtual_nodes: bool,
        ipr_virtual_message_layers: int,
        use_sh_branch: bool,
        use_cross_space: bool,
        use_cross_space_gate: bool,
        use_cross_space_semantic: bool,
        return_sh_cutoff: float,
        return_max_neighbors: int = None,
    ):
        super().__init__()
        self.dualequi = ResidueDualEquiLayer(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            num_heads=num_heads,
            lmax=lmax,
            dropout=0.0,
            use_sh_branch=use_sh_branch,
            use_cross_space=use_cross_space,
            use_cross_space_gate=use_cross_space_gate,
            use_cross_space_semantic=use_cross_space_semantic,
        )
        self.ipr_context = (
            IPRVirtualNodeRefiner(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                dropout=dropout,
                message_layers=ipr_virtual_message_layers,
            )
            if use_ipr_virtual_nodes
            else None
        )

    def forward(
        self,
        residue_h: torch.Tensor,
        residue_pos: torch.Tensor,
        residue_sh: torch.Tensor,
        residue_edge_index_eu: torch.Tensor,
        residue_edge_attr_eu: torch.Tensor,
        residue_edge_index_sh: torch.Tensor,
        state: IPRSharedState = None,
        virtual_h: torch.Tensor = None,
        virtual_pos: torch.Tensor = None,
        virtual_sh: torch.Tensor = None,
    ):
        local_h, local_pos, local_sh = self.dualequi(
            residue_h,
            residue_pos,
            residue_edge_index_eu,
            residue_edge_attr_eu,
            residue_edge_index_sh,
            residue_sh,
        )
        if self.ipr_context is not None:
            if state is None:
                raise ValueError("IPRSharedState is required when use_ipr_virtual_nodes=True.")
            residue_h, residue_pos, state = self.ipr_context(local_h, local_pos, state)
            virtual_h = state.virtual_x
            virtual_pos = state.virtual_pos
            virtual_sh = state.assignment.t().matmul(local_sh) / state.assignment.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
            residue_sh = local_sh
        else:
            residue_h, residue_pos, residue_sh = local_h, local_pos, local_sh
        return residue_h, residue_pos, residue_sh, state, virtual_h, virtual_pos, virtual_sh


class ResidueDualEquiEncoderBase(nn.Module):
    def _pool_virtual_sh(self, residue_sh: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        denom = assignment.sum(dim=0).clamp_min(1e-12).unsqueeze(-1)
        return assignment.t().matmul(residue_sh) / denom

    def _zero_edge_attr(self, num_edges: int, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros((num_edges, self.edge_attr_dim))

    def _build_virtual_eu_graph(
        self,
        state: IPRSharedState,
        num_residues: int,
        like_attr: torch.Tensor,
    ):
        device = like_attr.device
        edge_parts = []
        attr_parts = []

        if state.residue_to_virtual_edge_index.numel() > 0:
            residue_src, virtual_dst = state.residue_to_virtual_edge_index
            edge_parts.append(torch.stack([virtual_dst + num_residues, residue_src], dim=0))
            attr_parts.append(self._zero_edge_attr(residue_src.numel(), like_attr))

        if state.virtual_to_residue_edge_index.numel() > 0:
            virtual_src, residue_dst = state.virtual_to_residue_edge_index
            edge_parts.append(torch.stack([residue_dst, virtual_src + num_residues], dim=0))
            attr_parts.append(self._zero_edge_attr(virtual_src.numel(), like_attr))

        if state.virtual_to_virtual_edge_index.numel() > 0:
            virtual_src, virtual_dst = state.virtual_to_virtual_edge_index
            edge_parts.append(torch.stack([virtual_dst + num_residues, virtual_src + num_residues], dim=0))
            attr_parts.append(self._zero_edge_attr(virtual_src.numel(), like_attr))

        if not edge_parts:
            return _empty_edge_index(device), self._zero_edge_attr(0, like_attr)
        return torch.cat(edge_parts, dim=1), torch.cat(attr_parts, dim=0)

    def _build_combined_eu_graph(
        self,
        residue_edge_index_eu: torch.Tensor,
        residue_edge_attr_eu: torch.Tensor,
        state: IPRSharedState,
        num_residues: int,
    ):
        virtual_edge_index, virtual_edge_attr = self._build_virtual_eu_graph(
            state,
            num_residues=num_residues,
            like_attr=residue_edge_attr_eu,
        )
        if residue_edge_index_eu.numel() == 0:
            return virtual_edge_index, virtual_edge_attr
        if virtual_edge_index.numel() == 0:
            return residue_edge_index_eu, residue_edge_attr_eu
        return (
            torch.cat([residue_edge_index_eu, virtual_edge_index], dim=1),
            torch.cat([residue_edge_attr_eu, virtual_edge_attr], dim=0),
        )

    def _refresh_eu_graph(
        self,
        residue_pos: torch.Tensor,
        residue_edge_attr_eu: torch.Tensor,
    ):
        if not self.residue_update_eu_neighbors:
            return self.residue_edge_index_eu, residue_edge_attr_eu
        residue_edge_index_eu = _radius_graph_single(
            residue_pos,
            cutoff=self.residue_eu_cutoff,
            max_neighbors=self.residue_max_neighbors,
        )
        residue_edge_attr_eu = residue_pos.new_zeros((residue_edge_index_eu.size(1), self.edge_attr_dim))
        return residue_edge_index_eu, residue_edge_attr_eu

    def _init_residue_sh(self, residue_h: torch.Tensor, residue_pos: torch.Tensor, residue_edge_index_eu: torch.Tensor):
        if self.uses_spherical_features:
            residue_sh = self.sh_init(residue_h, residue_pos, residue_edge_index_eu)
            residue_edge_index_sh = self.residue_sh_graph(residue_sh)
            return residue_sh, residue_edge_index_sh
        return residue_h.new_zeros((residue_h.size(0), 1)), _empty_edge_index(residue_h.device)

    def _refresh_sh_graph(self, residue_sh: torch.Tensor, residue_edge_index_sh: torch.Tensor):
        if self.uses_spherical_features and self.residue_update_sh_neighbors:
            return self.residue_sh_graph(residue_sh)
        return residue_edge_index_sh


class ResidueDualEquiEncoder(ResidueDualEquiEncoderBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_attr_dim: int,
        num_layers: int,
        dropout: float,
        residue_num_heads: int,
        residue_lmax: int,
        residue_sh_cutoff: float,
        residue_update_sh_neighbors: bool,
        residue_update_eu_neighbors: bool,
        residue_use_sh_branch: bool,
        residue_use_cross_space: bool,
        residue_cross_space_gate: bool,
        residue_cross_space_semantic: bool,
        residue_eu_cutoff: float,
        residue_max_neighbors: int = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.residue_update_sh_neighbors = bool(residue_update_sh_neighbors)
        self.residue_update_eu_neighbors = bool(residue_update_eu_neighbors)
        self.residue_eu_cutoff = float(residue_eu_cutoff)
        self.residue_max_neighbors = residue_max_neighbors
        self.uses_spherical_features = bool(residue_use_sh_branch)
        if self.uses_spherical_features:
            self.sh_init = ResidueSHInit(hidden_dim, residue_lmax, 0.0)
            self.residue_sh_graph = ResidueSphericalHarmonicGraph(residue_sh_cutoff, max_neighbors=residue_max_neighbors)
        self.blocks = nn.ModuleList(
            [
                ResidueDualEquiLayer(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_heads=residue_num_heads,
                    lmax=residue_lmax,
                    dropout=0.0,
                    use_sh_branch=residue_use_sh_branch,
                    use_cross_space=residue_use_cross_space,
                    use_cross_space_gate=residue_cross_space_gate,
                    use_cross_space_semantic=residue_cross_space_semantic,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        residue_h = self.input_proj(x)
        self.residue_edge_index_eu, residue_edge_attr_eu = _remove_self_loops(
            _to_receiver_first(edge_index),
            edge_attr,
        )
        residue_sh, residue_edge_index_sh = self._init_residue_sh(residue_h, pos, self.residue_edge_index_eu)

        for block in self.blocks:
            residue_h, pos, residue_sh = block(
                residue_h,
                pos,
                self.residue_edge_index_eu,
                residue_edge_attr_eu,
                residue_edge_index_sh,
                residue_sh,
            )
            self.residue_edge_index_eu, residue_edge_attr_eu = self._refresh_eu_graph(pos, residue_edge_attr_eu)
            residue_edge_index_sh = self._refresh_sh_graph(residue_sh, residue_edge_index_sh)
        return self.norm(residue_h), pos


class ResidueDualEquiHierarchicalEncoder(ResidueDualEquiEncoderBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_attr_dim: int,
        num_layers: int,
        dropout: float,
        use_ipr_virtual_nodes: bool,
        ipr_num_virtual_nodes: int,
        ipr_temperature: float,
        ipr_sinkhorn_iterations: int,
        ipr_eval_hard: bool,
        ipr_scorer_context_layers: int,
        ipr_virtual_message_layers: int,
        ipr_aux_entropy_weight: float,
        ipr_aux_balance_weight: float,
        ipr_aux_compact_weight: float,
        ipr_aux_smooth_weight: float,
        ipr_aux_score_l2_weight: float,
        residue_num_heads: int,
        residue_lmax: int,
        residue_sh_cutoff: float,
        residue_update_sh_neighbors: bool,
        residue_update_eu_neighbors: bool,
        residue_use_sh_branch: bool,
        residue_use_cross_space: bool,
        residue_cross_space_gate: bool,
        residue_cross_space_semantic: bool,
        residue_eu_cutoff: float,
        residue_max_neighbors: int = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ipr_score_proj = nn.Linear(input_dim, hidden_dim) if use_ipr_virtual_nodes else None
        self.edge_attr_dim = int(edge_attr_dim)
        self.residue_update_sh_neighbors = bool(residue_update_sh_neighbors)
        self.residue_update_eu_neighbors = bool(residue_update_eu_neighbors)
        self.residue_eu_cutoff = float(residue_eu_cutoff)
        self.residue_max_neighbors = residue_max_neighbors
        self.uses_spherical_features = bool(residue_use_sh_branch)
        if self.uses_spherical_features:
            self.sh_init = ResidueSHInit(hidden_dim, residue_lmax, 0.0)
            self.residue_sh_graph = ResidueSphericalHarmonicGraph(residue_sh_cutoff, max_neighbors=residue_max_neighbors)

        self.ipr_sampler = (
            IPRVirtualNodeContext(
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                num_centroids=ipr_num_virtual_nodes,
                temperature=ipr_temperature,
                sinkhorn_iterations=ipr_sinkhorn_iterations,
                eval_hard=ipr_eval_hard,
                scorer_context_layers=ipr_scorer_context_layers,
                aux_balance_weight=ipr_aux_balance_weight,
                aux_entropy_weight=ipr_aux_entropy_weight,
                aux_compact_weight=ipr_aux_compact_weight,
                aux_smooth_weight=ipr_aux_smooth_weight,
                aux_score_l2_weight=ipr_aux_score_l2_weight,
                dropout=dropout,
            )
            if use_ipr_virtual_nodes
            else None
        )
        self.blocks = nn.ModuleList(
            [
                ResidueDualEquiIPRBlock(
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_heads=residue_num_heads,
                    lmax=residue_lmax,
                    dropout=dropout,
                    use_ipr_virtual_nodes=use_ipr_virtual_nodes,
                    ipr_virtual_message_layers=ipr_virtual_message_layers,
                    use_sh_branch=residue_use_sh_branch,
                    use_cross_space=residue_use_cross_space,
                    use_cross_space_gate=residue_cross_space_gate,
                    use_cross_space_semantic=residue_cross_space_semantic,
                    return_sh_cutoff=residue_sh_cutoff,
                    return_max_neighbors=residue_max_neighbors,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_aux: bool = False,
    ):
        raw_x = x
        residue_h = self.input_proj(raw_x)
        self.residue_edge_index_eu, residue_edge_attr_eu = _remove_self_loops(
            _to_receiver_first(edge_index),
            edge_attr,
        )
        residue_sh, residue_edge_index_sh = self._init_residue_sh(residue_h, pos, self.residue_edge_index_eu)

        aux_loss = residue_h.new_zeros(())
        state = None
        virtual_h = None
        virtual_pos = None
        virtual_sh = None
        if self.ipr_sampler is not None:
            score_h = self.ipr_score_proj(raw_x)
            if return_aux:
                state, aux_loss = self.ipr_sampler(
                    residue_h,
                    pos,
                    edge_index,
                    edge_attr,
                    return_aux=True,
                    score_x=score_h,
                )
            else:
                state = self.ipr_sampler(residue_h, pos, edge_index, edge_attr, score_x=score_h)
            virtual_h = state.virtual_x
            virtual_pos = state.virtual_pos
            virtual_sh = self._pool_virtual_sh(residue_sh, state.assignment)
        for block in self.blocks:
            residue_h, pos, residue_sh, state, virtual_h, virtual_pos, virtual_sh = block(
                residue_h,
                pos,
                residue_sh,
                self.residue_edge_index_eu,
                residue_edge_attr_eu,
                residue_edge_index_sh,
                state=state,
                virtual_h=virtual_h,
                virtual_pos=virtual_pos,
                virtual_sh=virtual_sh,
            )
            self.residue_edge_index_eu, residue_edge_attr_eu = self._refresh_eu_graph(pos, residue_edge_attr_eu)
            residue_edge_index_sh = self._refresh_sh_graph(residue_sh, residue_edge_index_sh)
        return self.norm(residue_h), pos, aux_loss

    def ipr_warmup_loss(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        if self.ipr_sampler is None:
            residue_h = self.input_proj(x)
            return residue_h.new_zeros(()), {}
        residue_h = self.input_proj(x)
        score_h = self.ipr_score_proj(x)
        _, aux_loss, diagnostics = self.ipr_sampler(
            residue_h,
            pos,
            edge_index,
            edge_attr,
            return_aux=True,
            force_soft_assignment=True,
            return_diagnostics=True,
            score_x=score_h,
        )
        return aux_loss, diagnostics


class PPISModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        egnn_layers: int,
        dropout: float,
        use_ipr_virtual_nodes: bool = False,
        ipr_num_virtual_nodes: int = 2,
        ipr_temperature: float = 1.0,
        ipr_sinkhorn_iterations: int = 20,
        ipr_eval_hard: bool = False,
        ipr_scorer_context_layers: int = 1,
        ipr_virtual_message_layers: int = 2,
        ipr_aux_entropy_weight: float = 0.001,
        ipr_aux_balance_weight: float = 0.001,
        ipr_aux_compact_weight: float = 0.01,
        ipr_aux_smooth_weight: float = 0.01,
        ipr_aux_score_l2_weight: float = 0.0005,
        encoder_type: str = "egnn",
        residue_num_heads: int = 4,
        residue_lmax: int = 2,
        residue_sh_cutoff: float = 0.95,
        residue_update_sh_neighbors: bool = True,
        residue_update_eu_neighbors: bool = False,
        residue_use_sh_branch: bool = True,
        residue_use_cross_space: bool = True,
        residue_cross_space_gate: bool = False,
        residue_cross_space_semantic: bool = False,
        residue_eu_cutoff: float = 14.0,
        residue_max_neighbors: int = None,
    ):
        super().__init__()
        if encoder_type not in {"egnn", "residue_dualequi"}:
            raise ValueError("encoder_type must be 'egnn' or 'residue_dualequi'.")

        self.encoder_type = encoder_type
        self.use_ipr_virtual_nodes = bool(use_ipr_virtual_nodes)
        if encoder_type == "residue_dualequi":
            if self.use_ipr_virtual_nodes:
                self.egnn = ResidueDualEquiHierarchicalEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_layers=egnn_layers,
                    dropout=dropout,
                    use_ipr_virtual_nodes=use_ipr_virtual_nodes,
                    ipr_num_virtual_nodes=ipr_num_virtual_nodes,
                    ipr_temperature=ipr_temperature,
                    ipr_sinkhorn_iterations=ipr_sinkhorn_iterations,
                    ipr_eval_hard=ipr_eval_hard,
                    ipr_scorer_context_layers=ipr_scorer_context_layers,
                    ipr_virtual_message_layers=ipr_virtual_message_layers,
                    ipr_aux_entropy_weight=ipr_aux_entropy_weight,
                    ipr_aux_balance_weight=ipr_aux_balance_weight,
                    ipr_aux_compact_weight=ipr_aux_compact_weight,
                    ipr_aux_smooth_weight=ipr_aux_smooth_weight,
                    ipr_aux_score_l2_weight=ipr_aux_score_l2_weight,
                    residue_num_heads=residue_num_heads,
                    residue_lmax=residue_lmax,
                    residue_sh_cutoff=residue_sh_cutoff,
                    residue_update_sh_neighbors=residue_update_sh_neighbors,
                    residue_update_eu_neighbors=residue_update_eu_neighbors,
                    residue_use_sh_branch=residue_use_sh_branch,
                    residue_use_cross_space=residue_use_cross_space,
                    residue_cross_space_gate=residue_cross_space_gate,
                    residue_cross_space_semantic=residue_cross_space_semantic,
                    residue_eu_cutoff=residue_eu_cutoff,
                    residue_max_neighbors=residue_max_neighbors,
                )
            else:
                self.egnn = ResidueDualEquiEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_layers=egnn_layers,
                    dropout=dropout,
                    residue_num_heads=residue_num_heads,
                    residue_lmax=residue_lmax,
                    residue_sh_cutoff=residue_sh_cutoff,
                    residue_update_sh_neighbors=residue_update_sh_neighbors,
                    residue_update_eu_neighbors=residue_update_eu_neighbors,
                    residue_use_sh_branch=residue_use_sh_branch,
                    residue_use_cross_space=residue_use_cross_space,
                    residue_cross_space_gate=residue_cross_space_gate,
                    residue_cross_space_semantic=residue_cross_space_semantic,
                    residue_eu_cutoff=residue_eu_cutoff,
                    residue_max_neighbors=residue_max_neighbors,
                )
        elif self.use_ipr_virtual_nodes:
            self.egnn = EGNNHierarchicalEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                num_layers=egnn_layers,
                dropout=dropout,
                use_ipr_virtual_nodes=use_ipr_virtual_nodes,
                ipr_num_virtual_nodes=ipr_num_virtual_nodes,
                ipr_temperature=ipr_temperature,
                ipr_sinkhorn_iterations=ipr_sinkhorn_iterations,
                ipr_eval_hard=ipr_eval_hard,
                ipr_scorer_context_layers=ipr_scorer_context_layers,
                ipr_virtual_message_layers=ipr_virtual_message_layers,
                ipr_aux_entropy_weight=ipr_aux_entropy_weight,
                ipr_aux_balance_weight=ipr_aux_balance_weight,
                ipr_aux_compact_weight=ipr_aux_compact_weight,
                ipr_aux_smooth_weight=ipr_aux_smooth_weight,
                ipr_aux_score_l2_weight=ipr_aux_score_l2_weight,
            )
        else:
            self.egnn = EGNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim,
                num_layers=egnn_layers,
                dropout=dropout,
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_aux: bool = False,
    ):
        if self.use_ipr_virtual_nodes:
            node_embeddings, _, aux_loss = self.egnn(
                node_features,
                pos,
                edge_index,
                edge_attr,
                return_aux=return_aux,
            )
        else:
            node_embeddings, _ = self.egnn(node_features, pos, edge_index, edge_attr)
            aux_loss = node_embeddings.new_zeros(())

        logits = self.classifier(node_embeddings).squeeze(-1)
        if return_aux:
            return logits, aux_loss
        return logits

    def ipr_warmup_loss(
        self,
        node_features: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        if not self.use_ipr_virtual_nodes or not hasattr(self.egnn, "ipr_warmup_loss"):
            return node_features.new_zeros(()), {}
        return self.egnn.ipr_warmup_loss(node_features, pos, edge_index, edge_attr)
