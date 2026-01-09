import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm
from .sampler import Sampler


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self._denoise_fn = denoise_fn
        self.sampling_batch_size = H.sampling_batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        self.sample_schedule = H.sample_schedule
        self.register_buffer('mask_id', torch.tensor(mask_id))

        # Partial masking strategy
        self.masking_strategy = getattr(H, 'masking_strategy', None)

        # Track loss at each time step for importance sampling
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))
        assert self.mask_schedule in ['random', 'fixed']

        self.task_queue = []

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            # get its probability
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # Randomly set *tokens* to mask with probability t/T.
        #
        # For Octuple (or Trio/Melody), we interpret a "token" as one timestep in the
        # sequence, i.e. we mask all channels at that position together.
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        b, seq_len = x_t.shape[0], x_t.shape[1]
        device = x_t.device

        # mask positions (B, L) with prob t/T per sample
        time_prob = (t.float() / self.num_timesteps).view(-1, 1)
        mask_pos = torch.rand((b, seq_len), device=device) < time_prob

        # expand to all channels (B, L, C)
        mask = mask_pos.unsqueeze(-1).expand_as(x_t)

        for i in range(len(self.mask_id)):
            x_t[:, :, i][mask_pos] = self.mask_id[i]

        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def q_sample_partial(self, x_0, t):
        """
        Implementation of partial masking strategies for Octuple MIDI.
        x_0 shape: (Batch, SeqLen, 8)
        """
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        b, seq_len, _ = x_0.shape
        device = x_0.device

        # Initialize mask with False
        mask = torch.zeros_like(x_t, dtype=torch.bool, device=device)
        
        # Time-based probability of masking: t/T
        time_prob = t.float() / self.num_timesteps # (Batch,)
        
        # Strategy Implementation
        current_strategy = self.masking_strategy

        if current_strategy == 'mixed':
            # Randomly select a strategy from the pool
            # Pool includes the 5 partial strategies AND 'random' (standard full masking)
            strategies = [
                '1_bar_all', 
                '2_bar_all', 
                '1_bar_attribute', 
                '2_bar_attribute', 
                'rand_attribute',
                'random'
            ]
            # Select one strategy for the entire batch (simplest implementation)
            # OR per sample? Per batch is standard for training stability.
            current_strategy = strategies[np.random.randint(len(strategies))]
        
        # If 'random' is selected (either explicitly or via mixed), use standard q_sample
        # (token-level/octuple masking).
        if current_strategy == 'random':
            return self.q_sample(x_0=x_0, t=t)

        if current_strategy == '1_bar_all':
            # Mask Pitch, Duration, Velocity, Tempo over 1 bar
            # Randomly select a bar for each sample
            # Bar info is in index 0
            
            # 1. Identify valid bars for each sample (assuming bars are sequential/present)
            # Simplification: Assume max bar from data or iterate. 
            # Since x_0 is a batch, we need to handle per-sample logic or vectorize.
            # Vectorized approach:
            
            # Find unique bars per sample is expensive. 
            # Let's assume standard Octuple structure where [:, :, 0] are bar indices.
            bar_indices = x_0[:, :, 0]

            # Select an existing bar per sample by picking a random position in the sequence
            rand_pos = torch.randint(0, seq_len, (b,), device=device)
            target_bars = bar_indices[torch.arange(b, device=device), rand_pos]
            
            # Expand for broadcasting
            target_bars_exp = target_bars.unsqueeze(1).expand(-1, seq_len) # (Batch, SeqLen)
            
            # Construct candidate mask: match bar and attributes
            # Attributes to mask: Pitch(3), Duration(4), Velocity(5), Tempo(7)
            target_attributes = torch.tensor([3, 4, 5, 7], device=device)
            
            # Create attribute mask
            attr_mask = torch.zeros(8, dtype=torch.bool, device=device)
            attr_mask[target_attributes] = True
            attr_mask = attr_mask.unsqueeze(0).unsqueeze(0).expand(b, seq_len, -1) # (Batch, SeqLen, 8)
            
            # Combine: (Bar match) AND (Attribute match)
            candidate_mask = (bar_indices == target_bars_exp).unsqueeze(-1) & attr_mask

        elif current_strategy == '2_bar_all':
            # Mask Pitch, Duration, Velocity, Tempo over 2 bars
            bar_indices = x_0[:, :, 0]

            # Pick two (possibly identical) existing bars per sample
            rand_pos1 = torch.randint(0, seq_len, (b,), device=device)
            rand_pos2 = torch.randint(0, seq_len, (b,), device=device)
            r1 = bar_indices[torch.arange(b, device=device), rand_pos1]
            r2 = bar_indices[torch.arange(b, device=device), rand_pos2]
            
            target_bars1 = r1.unsqueeze(1).expand(-1, seq_len)
            target_bars2 = r2.unsqueeze(1).expand(-1, seq_len)
            
            target_attributes = torch.tensor([3, 4, 5, 7], device=device)
            attr_mask = torch.zeros(8, dtype=torch.bool, device=device)
            attr_mask[target_attributes] = True
            attr_mask = attr_mask.unsqueeze(0).unsqueeze(0).expand(b, seq_len, -1)
            
            bar_match = (bar_indices == target_bars1) | (bar_indices == target_bars2)
            candidate_mask = bar_match.unsqueeze(-1) & attr_mask

        elif current_strategy == '1_bar_attribute':
            # Randomly pick ONE of {Pitch, Dur, Vel, Tempo} and mask it for 1 random bar
            # Per sample defined strategy
            
            bar_indices = x_0[:, :, 0]

            rand_pos = torch.randint(0, seq_len, (b,), device=device)
            target_bars = bar_indices[torch.arange(b, device=device), rand_pos]
            target_bars_exp = target_bars.unsqueeze(1).expand(-1, seq_len)
            
            # Available attributes: 3, 4, 5, 7
            avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
            # Pick one index per sample from 0..3
            rand_indices = (torch.rand(b, device=device) * 4).long()
            selected_attrs = avail_attrs[rand_indices] # (Batch,)
            
            selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1) # (Batch, SeqLen, 1)
            # Create full index tensor for last dim to compare
            feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
            
            attr_mask = (feature_indices == selected_attrs_exp)
            
            candidate_mask = (bar_indices == target_bars_exp).unsqueeze(-1) & attr_mask

        elif current_strategy == '2_bar_attribute':
            # Randomly pick ONE of {Pitch, Dur, Vel, Tempo} and mask it for 2 random bars
            
            bar_indices = x_0[:, :, 0]

            rand_pos1 = torch.randint(0, seq_len, (b,), device=device)
            rand_pos2 = torch.randint(0, seq_len, (b,), device=device)
            r1 = bar_indices[torch.arange(b, device=device), rand_pos1]
            r2 = bar_indices[torch.arange(b, device=device), rand_pos2]
            
            target_bars1 = r1.unsqueeze(1).expand(-1, seq_len)
            target_bars2 = r2.unsqueeze(1).expand(-1, seq_len)
            
            avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
            rand_indices = (torch.rand(b, device=device) * 4).long()
            selected_attrs = avail_attrs[rand_indices]
            
            selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
            feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
            
            attr_mask = (feature_indices == selected_attrs_exp)
            bar_match = (bar_indices == target_bars1) | (bar_indices == target_bars2)
            
            candidate_mask = bar_match.unsqueeze(-1) & attr_mask

        elif current_strategy == 'rand_attribute':
             # Randomly pick ONE of {Pitch, Dur, Vel, Tempo} and mask it randomly across the sequence
            
            avail_attrs = torch.tensor([3, 4, 5, 7], device=device)
            rand_indices = (torch.rand(b, device=device) * 4).long()
            selected_attrs = avail_attrs[rand_indices]
            
            selected_attrs_exp = selected_attrs.unsqueeze(1).expand(-1, seq_len).unsqueeze(-1)
            feature_indices = torch.arange(8, device=device).unsqueeze(0).unsqueeze(0).expand(b, seq_len, 8)
            
            candidate_mask = (feature_indices == selected_attrs_exp)

        else:
             # Fallback or error usage
             raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

        # Apply Time-dependence
        # "on the maximum t (1024) the probability of masking ... is 100% and it decreases until t=0... is 0%"
        # Effectively: Final_Mask = Candidate_Mask AND (Bernoulli(t/T))
        
        # Diffusion time gating (t/T), applied at the *unit* level.
        # This preserves "mask simultaneously" within each (bar, attribute) unit.
        gate = (torch.rand((b,), device=device) < time_prob).view(-1, 1, 1)

        # Combine
        mask = candidate_mask & gate

        # Apply per-channel mask token
        for i in range(len(self.mask_id)):
            x_t[:, :, i][mask[:, :, i]] = self.mask_id[i]
        
        x_0_ignore[torch.bitwise_not(mask)] = -1
        
        return x_t, x_0_ignore, mask

    def _train_loss(self, x_0):
        x_0 = x_0.long()
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise

        if self.masking_strategy is not None:
             x_t, x_0_ignore, mask = self.q_sample_partial(x_0=x_0, t=t)
        elif self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = self._denoise_fn(x_t)
        x_0_hat_logits = [el.permute(0, 2, 1) for el in x_0_hat_logits]

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = [F.cross_entropy(x, x_0_ignore[:, :, i], ignore_index=-1, reduction='none').sum(1)
                              for i, x in enumerate(x_0_hat_logits)]
        cross_entropy_loss = torch.stack(cross_entropy_loss).sum(0)

        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        return loss.mean(), vb_loss.mean()

    def sample(self, temp=1.0, sample_steps=None, x_T=None, B=None, progress_handler=None):
        b, device = self.sampling_batch_size, 'cuda'
        if B is not None:
            b = B
        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id
        b = x_T.shape[0]
        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))
        last_progress = 0

        for t in reversed(sample_steps):

            p = int(100 * (len(sample_steps) - t) / len(sample_steps))
            if progress_handler and p > last_progress:
                last_progress = p
                progress_handler(p)

            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_T.shape, device=device) < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = self._denoise_fn(x_T, t=t)
            # scale by temperature
            x_0_logits = [x / temp for x in x_0_logits]
            x_0_dist = [dists.Categorical(
                logits=x) for x in x_0_logits]
            x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)
            x_T[changes] = x_0_hat[changes]

        if progress_handler:
            progress_handler(100)

        return x_T

    def queue_sample_task(self, progress_handler, finished_handler, sample_steps=None, x_T=None, b=1):
        device = 'cuda'

        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id

        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))

        for tensor in x_T:
            self.task_queue.append((tensor, sample_steps, progress_handler, finished_handler))

    def sample_worker(self, temp=1.0):
        device = 'cuda'
        last_progress = 0

        x_T = torch.ones((0, *self.shape), device=device).long() * self.mask_id

        while True:

            while x_T.shape[0] < 1:

                x_T = torch.stack((x_T, ))

            for t in reversed(sample_steps):

                p = int(100 * (len(sample_steps) - t) / len(sample_steps))
                if progress_handler and p > last_progress:
                    last_progress = p
                    progress_handler(p)

                print(f'Sample timestep {t:4d}', end='\r')
                t = torch.full((b,), t, device=device, dtype=torch.long)

                # where to unmask
                changes = torch.rand(x_T.shape, device=device) < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)

                x_0_logits = self._denoise_fn(x_T, t=t)
                # scale by temperature
                x_0_logits = [x / temp for x in x_0_logits]
                x_0_dist = [dists.Categorical(
                    logits=x) for x in x_0_logits]
                x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)
                x_T[changes] = x_0_hat[changes]

        return x_T

    def guided_sample(self, guide, eta=1, temp=1.0, sample_steps=None, x_T=None, B=None):
        b, device = self.sampling_batch_size, 'cuda'
        if B is not None:
            b = B
        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id
        b = x_T.shape[0]


        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')

            with torch.no_grad():
                x_0_logits = self._denoise_fn(x_T, t=t)
            # scale by temperature
            x_0_logits = [x / temp for x in x_0_logits]
            x_0_probs = [F.softmax(x, -1) for x in x_0_logits]

            n = min(2, len(x_0_probs))

            for i in range(n): x_0_probs[i].requires_grad = True

            guide_loss = [guide(x_0_probs[i]) for i in range(n)]
            #print(guide_loss[0].item())
            #guide_loss = [g.log() for g in guide_loss] todo: logits require log?

            for i in range(n):
                guide_loss[i].mean().backward()
                #print("loss:", guide_loss[i].mean().item())

            #grads = torch.stack([x_0_p.grad.data[0, i, x_0_hat[0, i, 0]] for i in range(x_0_hat.shape[1])])#x_0_p.grad.data[0].gather(-1, x_0_hat[:, :, 0])#todo: repair for batches

            grad = [x_0_probs[i].grad.data for i in range(n)]
            x_0_probs = [F.softmax(x_0_logits[i], -1) for i in range(n)]

            for i in range(n):
                x_0_probs[i] -= grad[i] * eta #/(grad[i].max() - grad[i].min())
                x_0_probs[i] = x_0_probs[i].clamp(0, 1)

            x_0_dist = [dists.Categorical(probs=p) for p in x_0_probs]

            x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)

            # where to unmask
            t = torch.full((b,), t, device=device, dtype=torch.long)
            unmask_rand = torch.rand(x_T.shape, device=device)
            #unmask_rand[:, :, 0] = grads
            changes = unmask_rand < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))


            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_T[changes] = x_0_hat[changes]

        return x_T

    def sample_mlm(self, temp=1.0, sample_steps=None):
        b, device = self.sampling_batch_size, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        sample_steps = np.linspace(1, self.num_timesteps, num=sample_steps).astype(np.long)

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
            elbo += cross_entropy_loss / t
        return elbo

    def train_iter(self, x):
        loss, vb_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss}
        return stats

    @torch.no_grad()
    def sample_shape(self, shape, num_samples, time_steps=1000, step=1, temp=0.8):
        device = 'cuda'
        x_t = torch.ones((num_samples,) + shape, device=device).long() * self.mask_id
        x_lim = shape[0] - self.shape[0]

        unmasked = torch.zeros_like(x_t, device=device).bool()

        autoregressive_step = 0
        for t in tqdm(list(reversed(list(range(1, time_steps+1))))):
            t = torch.full((num_samples,), t, device='cuda', dtype=torch.long)

            unmasking_method = 'random'
            if unmasking_method == 'random':
                # where to unmask
                changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1).unsqueeze(-1)
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)
            elif unmasking_method == 'autoregressive':
                changes = torch.zeros(x_t.shape, device=device).bool()
                index = (int(autoregressive_step / shape[1]), autoregressive_step % shape[1])
                changes[:, index[0], index[1]] = True
                unmasked = torch.bitwise_or(unmasked, changes)
                autoregressive_step += 1

            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + self.codebook_size, device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Monte carlo approximate this instead
            for i in range(0, x_lim+1, step):
                # collect local noisy area
                x_t_part = x_t[:, i:i+self.shape[0]]

                # increment count
                count[:, i:i+self.shape[0]] += 1.0

                # flatten
                #x_t_part = x_t_part.reshape(x_t_part.size(0), -1)

                # denoise
                x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                # unflatten
                #x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], -1)

                # multiply probabilities
                # for mixture
                x_0_probs[:, i:i+self.shape[0], 0] += torch.softmax(x_0_logits_part[0], dim=-1)#todo: [0] list trio

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_t[changes] = x_0_hat[changes]

        return x_t.cpu().numpy()
