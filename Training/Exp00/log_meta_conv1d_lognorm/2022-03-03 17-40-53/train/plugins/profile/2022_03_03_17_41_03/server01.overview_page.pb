?	?X?O?@@?X?O?@@!?X?O?@@	?&???5@?&???5@!?&???5@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?X?O?@@?j{?@@1?N?j#@A???????I?;p??)@Y??{h?@*	bX94??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?.9?D@!?N??o?X@)?.9?D@1?N??o?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???G6W@!?eZ???X@)??J#f???1?"?pG???:Preprocessing2F
Iterator::Model@?&M?b@!      Y@)f??
???1??4K? ??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapq?;J@!???b??X@)????m3u?1a?]?J???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 21.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?39.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t10.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?&???5@IV?H???H@Q??Zg??<@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?j{?@@?j{?@@!?j{?@@      ??!       "	?N?j#@?N?j#@!?N?j#@*      ??!       2	??????????????!???????:	?;p??)@?;p??)@!?;p??)@B      ??!       J	??{h?@??{h?@!??{h?@R      ??!       Z	??{h?@??{h?@!??{h?@b      ??!       JGPUY?&???5@b qV?H???H@y??Zg??<@?"v
Kgradient_tape/model/time_distributed_10/conv1d_5/conv1d/Conv2DBackpropInputConv2DBackpropInput?????̸?!?????̸?0"s
Ggradient_tape/model/time_distributed/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?-?C???! ?x@?=??0"@
$model/time_distributed/conv1d/conv1dConv2D}OC?????!߀?%Sw??"w
Kgradient_tape/model/time_distributed_2/conv1d_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???Yy??!?? ?????0"-
IteratorGetNext/_1_Send{??tb???!???L???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?????!??8L`??0"w
Kgradient_tape/model/time_distributed_4/conv1d_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Dq????!p ???0"u
Jgradient_tape/model/time_distributed_8/conv1d_4/conv1d/Conv2DBackpropInputConv2DBackpropInput'??.Z???!4?d???0"E
)model/time_distributed_10/conv1d_5/conv1dConv2D]??)??!??????"D
(model/time_distributed_8/conv1d_4/conv1dConv2D5/?N??!??z????Q      Y@Y>???>@a>???>X@q???vz???yv??㫭??"?
host?Your program is HIGHLY input-bound because 21.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?39.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t10.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 