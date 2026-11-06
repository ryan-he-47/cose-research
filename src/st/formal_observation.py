"""2026-10-09：日志步只读记录文字/图形/圈选/控制符的GMM状态。"""
import torch


@torch.no_grad()
def distribution_sums(target, mu, sigma, pi, kinds):
    target, mu, sigma, pi = target.float(), mu.detach(), sigma.detach(), pi.detach()
    log_joint = (pi+1e-12).log() - sigma.log().sum(-1) - .5*((target[:,None]-mu)/sigma).square().sum(-1)
    posterior = log_joint.argmax(-1)
    prior = pi.argmax(-1)
    row = torch.arange(len(target), device=target.device)
    values = torch.stack((sigma[row,posterior].mean(-1),
        (mu[row,posterior]-target).square().mean(-1),
        (mu[row,prior]-target).square().mean(-1),
        (prior == posterior).float(), -(pi*(pi+1e-12).log()).sum(-1)),dim=-1)
    result = {}
    fields = ('posterior_sigma','posterior_mu_mse','top_pi_mu_mse','top_pi_matches_posterior','pi_entropy')
    for name, mask in (('text',kinds==0),('sketch',kinds==1),('selection',kinds==2),('control',kinds>=3)):
        count = int(mask.sum())
        if not count:
            continue
        result[f'gmm:{name}:count'] = count
        for field, value in zip(fields,values[mask].sum(0).tolist()):
            result[f'gmm:{name}:{field}'] = value
        histogram = torch.bincount(posterior[mask],minlength=pi.shape[-1]).tolist()
        for component, value in enumerate(histogram):
            result[f'gmm:{name}:component_{component}'] = value
    return result


def observation_report(sums):
    result = {}
    for bucket in ('text','sketch','selection','control'):
        prefix = f'gmm:{bucket}:'
        count = sums.get(prefix+'count',0)
        if count:
            result[bucket] = {'count':int(count), **{
                name[len(prefix):]:value/count for name,value in sums.items()
                if name.startswith(prefix) and not name.startswith(prefix+'component_') and name != prefix+'count'},
                'posterior_components':{name[len(prefix+'component_'):]:int(value)
                    for name,value in sums.items() if name.startswith(prefix+'component_')}}
    return result
