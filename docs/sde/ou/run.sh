#!/bin/bash

for dt in 0.005 0.001;
do
	for p_lr in 1 0.5 0.1 0.05 0.01 0.005 0.001;
		do
			for vgp_lr in 0.005;
			do
				for vgp_x0_lr in 0.1;
				do
					nohup python ou_comparison.py -dir "data/33" -wandb_username NAME -log True -data_sites_lr 0. -vgp_lr $vgp_lr -vgp_x0_lr $vgp_x0_lr -o vgp_learning_$dt_$vgp_lr_$vgp_x0_lr -dt $dt -l True -d -0.2 -prior_vgp_lr $p_lr &
					sleep 20
				done
			done
		done
	done
done
