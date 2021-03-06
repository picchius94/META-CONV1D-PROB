?	?ݑ?ZSb@?ݑ?ZSb@!?ݑ?ZSb@	!s[?*@!s[?*@!!s[?*@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ݑ?ZSb@??ۻ?\@1?`??>?"@AxADj????I????;-@Y^?V$&@*sh????@)      p=2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator???8?@!$?8???X@)???8?@1$?8???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismݗ3۵@!????J?X@)?#??????1W"??????:Preprocessing2F
Iterator::Model8???L?@!      Y@)#?dT?}?1?ȥ?]պ?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap???c[?@!|K??R?X@)-|}?K?p?1޿*??խ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t78.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9 s[?*@I)P?vlV@QP?υx?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ۻ?\@??ۻ?\@!??ۻ?\@      ??!       "	?`??>?"@?`??>?"@!?`??>?"@*      ??!       2	xADj????xADj????!xADj????:	????;-@????;-@!????;-@B      ??!       J	^?V$&@^?V$&@!^?V$&@R      ??!       Z	^?V$&@^?V$&@!^?V$&@b      ??!       JGPUY s[?*@b q)P?vlV@yP?υx?@?"y
Ngradient_tape/model_1/time_distributed_25/conv1d_11/conv1d/Conv2DBackpropInputConv2DBackpropInputgw??ʸ?!gw??ʸ?0"z
Ngradient_tape/model_1/time_distributed_15/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter td???!4?u??p??0"G
+model_1/time_distributed_15/conv1d_6/conv1dConv2D_???o???!LP`?+???"z
Ngradient_tape/model_1/time_distributed_17/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?*? ???!H????0"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropB	g-?[??!p`?????0"-
IteratorGetNext/_1_Send?;O?????!?Gl%???"z
Ngradient_tape/model_1/time_distributed_19/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilterx???????!????!;??0"y
Ngradient_tape/model_1/time_distributed_23/conv1d_10/conv1d/Conv2DBackpropInputConv2DBackpropInput??r?????!?7?zAo??0"H
,model_1/time_distributed_25/conv1d_11/conv1dConv2D4X?*??!h^?%????"H
,model_1/time_distributed_23/conv1d_10/conv1dConv2D??r????!???S????Q      Y@Y>???>@a>???>X@qk??I?>@yxƇn???"?
both?Your program is MODERATELY input-bound because 5.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t78.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?30.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 