?	?R???Ef@?R???Ef@!?R???Ef@	cuH+@cuH+@!cuH+@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?R???Ef@?fHE?a@1?dp??#@AF????x??ISAEկ1@Y???l&@*?~j?$??@)      ?=2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorG???+@!??0<??X@)G???+@1??0<??X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?X??+@!	&?ٵ?X@)?o??}??1?Ar~;???:Preprocessing2F
Iterator::Model?R#?3-+@!      Y@)qY?? ??1??n{??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???s(#+@!?H@s??X@)??c${?z?1!d??d??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t78.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9cuH+@I}C??V@Q???C?h@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?fHE?a@?fHE?a@!?fHE?a@      ??!       "	?dp??#@?dp??#@!?dp??#@*      ??!       2	F????x??F????x??!F????x??:	SAEկ1@SAEկ1@!SAEկ1@B      ??!       J	???l&@???l&@!???l&@R      ??!       Z	???l&@???l&@!???l&@b      ??!       JGPUYcuH+@b q}C??V@y???C?h@?"y
Ngradient_tape/model_1/time_distributed_25/conv1d_11/conv1d/Conv2DBackpropInputConv2DBackpropInputH?W9?Ÿ?!H?W9?Ÿ?0"z
Ngradient_tape/model_1/time_distributed_15/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Z??^N??!z????0"-
IteratorGetNext/_1_Send??u??!??(??L??"z
Ngradient_tape/model_1/time_distributed_17/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?"??p??!?J?`?4??0"G
+model_1/time_distributed_15/conv1d_6/conv1dConv2DgQ???"??!???????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropDt?????!???/{|??0"z
Ngradient_tape/model_1/time_distributed_19/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?:NHK??!?j??D???0"y
Ngradient_tape/model_1/time_distributed_23/conv1d_10/conv1d/Conv2DBackpropInputConv2DBackpropInput?:?????!3?CD???0"H
,model_1/time_distributed_25/conv1d_11/conv1dConv2DLp????!< ?\;??"H
,model_1/time_distributed_23/conv1d_10/conv1dConv2D?h^9???!W?_?&??Q      Y@Y?	g??@a???O8X@q??6??k3@y#??????"?
both?Your program is MODERATELY input-bound because 6.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t78.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?19.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 