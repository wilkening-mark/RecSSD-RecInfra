import os, sys

# RM2

num_epochs  = 1
num_batches = 400

rt_config_cpu = "--inference_only --inter_op_workers 1 --caffe2_net_type simple --enable_profiling "
rt_config_gpu = "--inference_only --inter_op_workers 1 --caffe2_net_type simple --use_gpu --enable_profiling "

model_config = "--model_type dlrm --arch_mlp_top \"128-64-1\" --arch_mlp_bot \"128-64-64\" --arch_sparse_feature_size 64 --arch_embedding_size \"50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000-50000\" --num_indices_per_lookup 120 --num_indices_per_lookup_fixed True --arch_interaction_op cat "

# Run once with batchsize 128
for k in range(1,4):
	n = 2**7
	dist_file = "\"./input/dist_{}_emb_j.log\" ".format(k)
	data_file = "\"ids_{}.pkl\"".format(k)
	print(dist_file)
	print(data_file)
	data_config = "--nepochs " + str(num_epochs) + " --num_batches " + str(num_batches) + " --mini_batch_size " + str(n) + " --max_mini_batch_size " + str(n) + " --data_generation \"synthetic\" --data_trace_file " + dist_file + "--data_file " + data_file

	cpu_command = "python dlrm_data.py " + rt_config_cpu + model_config + data_config
	gpu_command = "python dlrm_data.py " + rt_config_gpu + model_config + data_config

	print("--------------------Running (RM2) CPU Test with Batch Size " + str(n) +"--------------------\n")
	print(cpu_command)
	os.system(cpu_command)
	#print("--------------------Running (RM2) GPU Test with Batch Size " + str(n) +"--------------------\n")
	# print(gpu_command)
	#os.system(gpu_command)
