seq=mocap_240724_Take10
seq=mocap_240724_Take12
euler_dir=/home/lixin/mount/euler/Dynamic3DGaussians/GSTAR/${seq}
scratch_dir=/home/lixin/mount/scratch/lixin/GSTAR/${seq}

mkdir ${euler_dir}/PhysAvatar
cp -r ${scratch_dir}/PhysAvatar/smplx_params ${euler_dir}/PhysAvatar/