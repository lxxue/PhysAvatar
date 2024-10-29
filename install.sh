conda create -n physavatar python=3.10
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html
pip install tqdm Pillow wandb
pip install open3d trimesh pyrender PyOpenGL scipy robust-laplacian smplx human-body-prior
