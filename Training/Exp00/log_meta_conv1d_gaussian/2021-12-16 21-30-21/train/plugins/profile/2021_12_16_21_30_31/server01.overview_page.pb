?	#?qpiwe@#?qpiwe@!#?qpiwe@	?=?'?>@?=?'?>@!?=?'?>@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6#?qpiwe@]?wb?Va@1?3??"@A??????IȗP???%@Y????~)@*	? ?r8?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generatorӄ?'c?*@!?5?	L?X@)ӄ?'c?*@1?5?	L?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?ɧǦ*@!}??-?X@)?c???Ȕ?1??)?Hx??:Preprocessing2F
Iterator::Model???Z?*@!      Y@)??󬤅?1??JEF??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??f??*@!V?.|??X@)???X????1.??q2??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t80.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?=?'?>@I???Y?U@Qs?ow??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?wb?Va@]?wb?Va@!]?wb?Va@      ??!       "	?3??"@?3??"@!?3??"@*      ??!       2	????????????!??????:	ȗP???%@ȗP???%@!ȗP???%@B      ??!       J	????~)@????~)@!????~)@R      ??!       Z	????~)@????~)@!????~)@b      ??!       JGPUY?=?'?>@b q???Y?U@ys?ow??@?"y
Ngradient_tape/model_4/time_distributed_70/conv1d_29/conv1d/Conv2DBackpropInputConv2DBackpropInput?7g??θ?!?7g??θ?0"{
Ogradient_tape/model_4/time_distributed_60/conv1d_24/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??%???!㍓? ???0"-
IteratorGetNext/_1_SendF??}???!???f/???"{
Ogradient_tape/model_4/time_distributed_62/conv1d_25/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??u?/???!?????n??0"H
,model_4/time_distributed_60/conv1d_24/conv1dConv2D5???@??!Ӫ?????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??????!???{C???0"{
Ogradient_tape/model_4/time_distributed_64/conv1d_26/conv1d/Conv2DBackpropFilterConv2DBackpropFilteri???H??!Sn.?`???0"y
Ngradient_tape/model_4/time_distributed_68/conv1d_28/conv1d/Conv2DBackpropInputConv2DBackpropInput??t???!???ڎ2??0"H
,model_4/time_distributed_70/conv1d_29/conv1dConv2D?!??(??!8?R??W??"H
,model_4/time_distributed_68/conv1d_28/conv1dConv2D???F????!7?.?7i??Q      Y@Y?T?x?r
@a[=;n,X@q-????&@y????pɚ?"?
both?Your program is MODERATELY input-bound because 7.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t80.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 