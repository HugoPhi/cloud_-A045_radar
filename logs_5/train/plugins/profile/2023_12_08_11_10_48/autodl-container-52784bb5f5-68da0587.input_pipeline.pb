$	�s���2n@��0;o�r@Q����<@!�g]��c|@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'�g]��c|@aU��N��?1�=�4ez@IE,b�=>@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 	Q����<@18�*5{�u?I�k�}��<@r9*	x��/�~@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateb�G,�?!�EN̑�P@)�����@�?1��j��N@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�f��,@)kF��0�?1�a�R@J'@:Preprocessing2F
Iterator::Model�
b�k�?!��+��.@)��C��?1Y�}K�y @:Preprocessing2U
Iterator::Model::ParallelMapV27���0�?!cV\y�@)7���0�?1cV\y�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceq�q�t��?!`�}t�|@)q�q�t��?1`�}t�|@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�*5{��?!�Hz7@)�*5{��?1�Hz7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�;Fz�?!���L U@)�D����?1Q�f+�'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���]�?!ӣ�zQ@)�E|'f�h?1��W��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�12.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIG|��/)@Qwp� �U@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	aU��N��?U'e���?!aU��N��?	!       "$	@�nJej@�/��r@8�*5{�u?!�=�4ez@*	!       2	!       :$	V�pA�=@���6�E�?�k�}��<@!E,b�=>@B	!       J	!       R	!       Z	!       b	!       JGPUb qG|��/)@ywp� �U@