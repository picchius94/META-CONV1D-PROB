?	??D@??D@!??D@	??OI?5@??OI?5@!??OI?5@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??D@?wE??E@13Q???D#@A???l ??I:u??<0@Y????!@*	+??Υ?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator=I?f? @!y?5???X@)=I?f? @1y?5???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-??. @!bM??X@)S?k%t???1}???$???:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?,?$ @!?.???X@)?????1??ꌏ??:Preprocessing2F
Iterator::Model=??- 4 @!      Y@)i?????1?:??e??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 21.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t13.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??OI?5@In????K@Q<?_u?7@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?wE??E@?wE??E@!?wE??E@      ??!       "	3Q???D#@3Q???D#@!3Q???D#@*      ??!       2	???l ?????l ??!???l ??:	:u??<0@:u??<0@!:u??<0@B      ??!       J	????!@????!@!????!@R      ??!       Z	????!@????!@!????!@b      ??!       JGPUY??OI?5@b qn????K@y<?_u?7@?"-
IteratorGetNext/_1_SendOD5s&	??!OD5s&	??"v
Kgradient_tape/model/time_distributed_10/conv1d_5/conv1d/Conv2DBackpropInputConv2DBackpropInputr??SG???!??`??I??0"s
Ggradient_tape/model/time_distributed/conv1d/conv1d/Conv2DBackpropFilterConv2DBackpropFilter!??????!xb?? ??0"w
Kgradient_tape/model/time_distributed_2/conv1d_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?w.?4??!??0r'??0"@
$model/time_distributed/conv1d/conv1dConv2D??K????!(D??????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropg???Ԥ?!?a??0"w
Kgradient_tape/model/time_distributed_4/conv1d_2/conv1d/Conv2DBackpropFilterConv2DBackpropFilter\*Y???!??*I????0"u
Jgradient_tape/model/time_distributed_8/conv1d_4/conv1d/Conv2DBackpropInputConv2DBackpropInput??.gl??!???*???0"E
)model/time_distributed_10/conv1d_5/conv1dConv2D0RYy????!Bλ?~???"D
(model/time_distributed_8/conv1d_4/conv1dConv2DJ??5?3??!??: ????Q      Y@Y?Oq??@a?h?>?W@q G՘^b??y????Pޚ?"?
host?Your program is HIGHLY input-bound because 21.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?40.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"t13.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 