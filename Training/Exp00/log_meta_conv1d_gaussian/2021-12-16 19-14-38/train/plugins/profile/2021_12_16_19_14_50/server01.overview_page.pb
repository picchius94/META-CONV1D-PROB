?	?|?X%]d@?|?X%]d@!?|?X%]d@	j?????@j?????@!j?????@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?|?X%]d@öE?q`@1??{?9#@AB	3m????I???V?+@Y73??pj@*	? ?r???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??e	"@!??3?X@)??e	"@1??3?X@:Preprocessing2F
Iterator::Model???/J"@!      Y@)?7? ?x?1??!J?6??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?3?/."@!?w?V??X@)?+???t?1H???x??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapK?b??
"@!??T#?X@):?Y?Xh?1??>?
٠?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9j?????@I?۾??[V@Q(???D?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	öE?q`@öE?q`@!öE?q`@      ??!       "	??{?9#@??{?9#@!??{?9#@*      ??!       2	B	3m????B	3m????!B	3m????:	???V?+@???V?+@!???V?+@B      ??!       J	73??pj@73??pj@!73??pj@R      ??!       Z	73??pj@73??pj@!73??pj@b      ??!       JGPUYj?????@b q?۾??[V@y(???D?@?"y
Ngradient_tape/model_1/time_distributed_25/conv1d_11/conv1d/Conv2DBackpropInputConv2DBackpropInput2?????!2?????0"z
Ngradient_tape/model_1/time_distributed_15/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?????P??!?Rs?P???0"-
IteratorGetNext/_1_Send?yz????!f1????"z
Ngradient_tape/model_1/time_distributed_17/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilter2=?s8??!Y?n?C???0"G
+model_1/time_distributed_15/conv1d_6/conv1dConv2DK????O??!"??)9F??"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??????!?w@?+???0"z
Ngradient_tape/model_1/time_distributed_19/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilterA??<??!Z?xK?-??0"y
Ngradient_tape/model_1/time_distributed_23/conv1d_10/conv1d/Conv2DBackpropInputConv2DBackpropInput???}????!??1?0a??0"H
,model_1/time_distributed_25/conv1d_11/conv1dConv2D??L ??!4h
:???"H
,model_1/time_distributed_23/conv1d_10/conv1dConv2D[?e??w??!?w?/???Q      Y@Y?T?x?r
@a[=;n,X@q??#?M*@y??i7????"?
both?Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 