	v?e??UB@v?e??UB@!v?e??UB@	???/?4@???/?4@!???/?4@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6v?e??UB@???ި?
@1X???T #@AL?1?=B??IF\ ??/@Y?
???@*	?????@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?S?*#@!,?(???X@)?S?*#@1,?(???X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapR?b?#@!???,?X@)Քd????1SM?̎n??:Preprocessing2F
Iterator::Model?-???#@!      Y@)??????1?4y?$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism>?$@M#@!f?????X@)??g?,??1%??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 20.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?43.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s9.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9 ???/?4@IX7R02?J@Q0???k?9@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???ި?
@???ި?
@!???ި?
@      ??!       "	X???T #@X???T #@!X???T #@*      ??!       2	L?1?=B??L?1?=B??!L?1?=B??:	F\ ??/@F\ ??/@!F\ ??/@B      ??!       J	?
???@?
???@!?
???@R      ??!       Z	?
???@?
???@!?
???@b      ??!       JGPUY ???/?4@b qX7R02?J@y0???k?9@