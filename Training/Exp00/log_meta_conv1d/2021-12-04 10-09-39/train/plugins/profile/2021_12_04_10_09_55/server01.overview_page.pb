?	ĔH???=@ĔH???=@!ĔH???=@	g)??d?3@g)??d?3@!g)??d?3@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ĔH???=@??XP4@1??4"@AV?j-?B??I????8?%@Y?	??b?@*	`??"?D?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??????@!/5???X@)??????@1/5???X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap¥c?3?@!?:?d??X@)F\ ?K??1y?ȗ<??:Preprocessing2F
Iterator::Model-"???@!      Y@)?5Z?P??14D®]d??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Os?@!ޞ(???X@)@??>??1Ȭت??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?36.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t12.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9g)??d?3@I??$???H@Q??+a`>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??XP4@??XP4@!??XP4@      ??!       "	??4"@??4"@!??4"@*      ??!       2	V?j-?B??V?j-?B??!V?j-?B??:	????8?%@????8?%@!????8?%@B      ??!       J	?	??b?@?	??b?@!?	??b?@R      ??!       Z	?	??b?@?	??b?@!?	??b?@b      ??!       JGPUYg)??d?3@b q??$???H@y??+a`>@?"v
Kgradient_tape/model/time_distributed_10/conv1d_5/conv1d/Conv2DBackpropInputConv2DBackpropInput?qsH?	??!?qsH?	??0"s
Ggradient_tape/model/time_distributed/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter5?K????!??2J3x??0"-
IteratorGetNext/_1_Sendd??Fƽ??!???ۤ'??"w
Kgradient_tape/model/time_distributed_2/conv1d_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??4L???!????[???0"@
$model/time_distributed/conv1d/conv1dConv2D6?W??!??A?>???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropD??????!??5<?V??0"w
Kgradient_tape/model/time_distributed_4/conv1d_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter>?L?|??!PX??a???0"u
Jgradient_tape/model/time_distributed_8/conv1d_4/conv1d/Conv2DBackpropInputConv2DBackpropInput3?<?V{??!??????0"E
)model/time_distributed_10/conv1d_5/conv1dConv2D??A????!?F?E??"D
(model/time_distributed_8/conv1d_4/conv1dConv2D?d1GJB??!:?zG?6??Q      Y@Y?Oq??@a?h?>?W@q??))???yұ^??'??"?
both?Your program is MODERATELY input-bound because 19.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?36.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t12.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 