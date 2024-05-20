import torch
import torch.nn.functional as F


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, sigma: float, num_samples: int):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k
        
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                    torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                    / ctx.num_samples
                    / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def compute_topk_hidden_states(self, hidden_states, token_relevance, n_tokens):
    k = int(self.top_p * n_tokens)

    if self.topk_inference == "perturbated":
        indicators = PerturbedTopKFunction.apply(token_relevance, k, self.sigma, self.n_samples)
        topk_encoder_outputs = torch.einsum("b k d, b d c -> b k c", indicators, hidden_states)
    elif self.topk_inference == "hard":
        if self.training:        
            indicators = PerturbedTopKFunction.apply(token_relevance, k, self.sigma, self.n_samples)
            topk_encoder_outputs = torch.einsum("b k n, b n d -> b k d", indicators, hidden_states)
        else:
            _, indices = torch.topk(token_relevance, k=k, dim=-1, sorted=False)
            indices = torch.sort(indices, dim=-1).values
            indicators = F.one_hot(indices)
            indicators = F.pad(indicators, pad=(0, n_tokens - indicators.size(-1), 0, 0))
            # import pickle; pickle.dump(indicators.cpu(), open("indicators.pkl", "wb"))
            topk_encoder_outputs = torch.einsum("b k n, b n d -> b k d", indicators.float(), hidden_states)

    return topk_encoder_outputs, indicators.cpu()


def topk_hidden_states_selector(self, hidden_states, layer_outputs, idx, topk_idx, position_ids=None):
    n_tokens = hidden_states.size(1)
    if self.mult_layers_topk and (idx + 1)%self.num_topk_layers==0:
        if self.use_attention:
            token_relevance = torch.mean(layer_outputs[1].squeeze(), dim=(1))
            token_relevance = torch.mean(token_relevance, dim=(0)).unsqueeze(0)
        else:
            token_relevance = self.score[topk_idx](hidden_states)
            token_relevance = token_relevance.squeeze(-1)
        hidden_states, self.indicators = compute_topk_hidden_states(self, hidden_states, token_relevance, n_tokens)
        topk_idx += 1
    elif self.encoder_topk_layer == idx:
        
        if self.use_attention:
            token_relevance = torch.mean(layer_outputs[1].squeeze(), dim=(1))
            token_relevance = torch.mean(token_relevance, dim=(0)).unsqueeze(0)
        else:
            for cls_layer in self.cls_layers:
                hidden_states = cls_layer(hidden_states, attention_mask=None, layer_head_mask=None)[0]
            token_relevance = self.score(hidden_states)
            token_relevance = token_relevance.squeeze(-1)
            if self.save_scores:
                attn_tensor = torch.mean(layer_outputs[1].squeeze(), dim=(1))
                attn_tensor = torch.mean(attn_tensor, dim=(0))
                self.average_attn_scores = attn_tensor.cpu().detach().tolist()
                self.scorer_scores = token_relevance.squeeze(0).cpu().detach().tolist()
        
        hidden_states, self.indicators = compute_topk_hidden_states(self, hidden_states, token_relevance, n_tokens)

    position_ids = position_ids[:, :hidden_states.size(1)]
    return hidden_states, topk_idx, position_ids

