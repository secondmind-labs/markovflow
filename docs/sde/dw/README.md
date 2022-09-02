## Double Well Experiment

True SDE: `dx = 4 * x * (1 - x**2) dt + dB`.

Prior SDE: `dx = a * x * (c - x**2) dt + dB`.

## Generate data

```shell
python dw_generate_data.py -s 18 -q 1.0 -n 20 -v 0.01 -t0 0. -t1 20.0 -dt 0.01 -x0 1.
```

where,

s = seed value \
q = spectral density \
n = number of observations \
v = observation variance \
t0 = initial t0 value \
t1 = initial t1 value \
dt = time-step for simulating the SDE \
x0 = initial value

The output is saved in the directory, `data/{seed_value}`.

## Inference
``` shell
python dw_comparison.py -dir "data/18" -data_sites_lr 0.9 -vgp_lr 0.05 -vgp_x0_lr 0.05 -o inference_01 -dt 0.001 -a 3.0 -c 1.0
```

where,

dir = data directory \
data_sites_lr = learning-rate for the data-sites; setting this to `0` would skip performing inference/training for t-VGP. \
vgp_lr = learning-rate for VGP; setting this to `0` would skip performing inference/training for VGP. \
vgp_x0_lr = learning-rate for VGP initial state. \
o = output directory name, by default it is set to `inference` or `learning`. \
dt = time-step for inference, by default it is equal to the simulating time-step used for generating data. \
a = initial value for scaling for the prior SDE. \
c = initial value for well for the prior SDE. \
log = `True`, if log using wandb. By default it is `False`. \
wandb_username = `str`, by default it is set to `None`.

## Learning
``` shell
python dw_comparison.py -dir "data/18" -data_sites_lr 0.9 -vgp_lr 0.05 -vgp_x0_lr 0.05 -o learning_01 -dt 0.001 -a 3.0 -c 1.0 -l True -prior_ssm_lr 0.01 -prior_vgp_lr 0.01
```