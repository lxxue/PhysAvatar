data_dir=/home/lixin/mount/scratch/GSTAR
seq=mocap_240724_Take10
# seq=mocap_240724_Take12
# seq=mocap_240906_Take3
# seq=mocap_240906_Take8
# seq=mocap_241016_Take2
# seq=mocap_241016_Take4
save_name=${seq}
CUDA_LAUNCH_BLOCKING=1 python train_mesh_lbs_gstar.py \
 --save_name ${save_name} --downsample_view 1 --num_frames -1 --start_idx 0 --seq ${seq} \
 --obj_name mesh_first_frame.obj --cloth_name mesh_first_frame.obj --data_path ${data_dir}/${seq}  \
 --lr_means3D 0.00004 --lr_colors 0.0025 --lr_smplx 0 \
 --normal_weight 0.1 --iso_weight 20  --area_weight 50 --eq_faces_weight 1000 --collision_weight 0 \
 --exp_name ${seq}_with_lbs_with_cloth --smplx_gender male \
 --wandb --wandb_entity lxxue --wandb_name ${seq} --wandb_proj PhysAvatar
