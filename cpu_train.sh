docker run -it \
-v /home/ubuntu/uai-sdk/examples/mxnet/train/AlphaPig:/data \
-v /home/ubuntu/uai-sdk/examples/mxnet/train/AlphaPig/sgf_data:/data/data \
-v /home/ubuntu/uai-sdk/examples/mxnet/train/AlphaPig/logs:/data/output \
uhub.service.ucloud.cn/uaishare/cpu_uaitrain_ubuntu-14.04_python-2.7.6_mxnet-1.0.0:v1.0 \
/bin/bash -c "cd /data && /usr/bin/python /data/train_mxnet.py --model-prefix=siler_Alpha --work_dir=/data  --output_dir=/data/output"






