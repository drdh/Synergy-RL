## Low-Rank Modular Reinforcement Learning via Muscle Synergy

This repository is the official implementation of [Low-Rank Modular Reinforcement Learning via Muscle Synergy](https://openreview.net/pdf?id=zYc5FSxL6ar).

## Setup

```bash
pip install -r requirements.txt
```

## Training

```
python main.py 
--custom_xml environments/humanoids 
--actor_type transformer 
--critic_type transformer 
--grad_clipping_value 0.1 
--attention_layers 3 
--attention_heads 2 
--lr 0.0001 
--transformer_norm 1 
--attention_hidden_size 256 
--condition_decoder 1
--label SOLAR
--save_buffer 1
--save_freq 100000
--start_timesteps 2000
--enable_synergy_obs 1\
--enable_synergy_act 1\
--synergy_action_dim 1\
--d_model 1\
```

We were using [Sacred](https://github.com/IDSIA/sacred) and [WandB](https://wandb.ai) for experiment management. 
You can disable WandB by setting `--enable_wandb 0`.  

To train other environments, change the `--custom_xml` to `hoppers`, `walkers`, 
`walker_humanoids_hopper`.

## Acknowledgement

- The code is built on top of [Amorpheus](https://github.com/yobibyte/amorpheus) and [SMP](https://github.com/huangwl18/modular-rl) repository.
- Initial implementation of the transformers was taken from the [official Pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) and modified thereafter. 
