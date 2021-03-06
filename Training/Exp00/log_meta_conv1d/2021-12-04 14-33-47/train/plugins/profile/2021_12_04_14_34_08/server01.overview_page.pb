?	<L????j@<L????j@!<L????j@	?X?Ԅ?@?X?Ԅ?@!?X?Ԅ?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6<L????j@ڑ?;?9f@17????"@AސFN???I??Jvld,@Y???Q??$@*	?|?5?:?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??gy?,@!??????X@)??gy?,@1??????X@:Preprocessing2F
Iterator::Modell?<*??,@!      Y@)?9?!??1?`?!X???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismu???m?,@!P?S??X@)??^?????1??k	C??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??}8H?,@!\???i?X@)???X????1?????ͬ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?X?Ԅ?@I$D"!9?V@Q4e.??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ڑ?;?9f@ڑ?;?9f@!ڑ?;?9f@      ??!       "	7????"@7????"@!7????"@*      ??!       2	ސFN???ސFN???!ސFN???:	??Jvld,@??Jvld,@!??Jvld,@B      ??!       J	???Q??$@???Q??$@!???Q??$@R      ??!       Z	???Q??$@???Q??$@!???Q??$@b      ??!       JGPUY?X?Ԅ?@b q$D"!9?V@y4e.??@?"y
Ngradient_tape/model_1/time_distributed_25/conv1d_11/conv1d/Conv2DBackpropInputConv2DBackpropInput?6??????!?6??????0"-
IteratorGetNext/_1_Send?K;?Э??!8A???I??"z
Ngradient_tape/model_1/time_distributed_15/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter*???덲?!f???iH??0"z
Ngradient_tape/model_1/time_distributed_17/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilterI;?s???!Ϧ?I8[??0"G
+model_1/time_distributed_15/conv1d_6/conv1dConv2D????/:??!̘eB~??"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop'?JZ??!???)???0"z
Ngradient_tape/model_1/time_distributed_19/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??H??G??!???)???0"y
Ngradient_tape/model_1/time_distributed_23/conv1d_10/conv1d/Conv2DBackpropInputConv2DBackpropInput?g?5???!?|?1j??0"H
,model_1/time_distributed_25/conv1d_11/conv1dConv2D???B??!?/y??@??"H
,model_1/time_distributed_23/conv1d_10/conv1dConv2D??3?m??!??O?'???Q      Y@Y?Oq??@a?h?>?W@q?'O?E@y!?????"?
both?Your program is POTENTIALLY input-bound because 83.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?42.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 