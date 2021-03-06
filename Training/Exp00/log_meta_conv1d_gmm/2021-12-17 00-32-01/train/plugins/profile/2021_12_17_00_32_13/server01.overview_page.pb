?	?7L4?Cd@?7L4?Cd@!?7L4?Cd@	
?"|???
?"|???!
?"|???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?7L4?Cd@f?B??a@1?v?5#@Am?kA???IA???F?"@Y^??Nw?@*	?v??r?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator_?vj.?@!h?K)νX@)_?vj.?@1h?K)νX@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap*?=%?@!1,D̶?X@)?e?c]ܖ?1[?9|Q???:Preprocessing2F
Iterator::Modelc'?'@!      Y@)??-?R??1??s0}b??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismi5$??@!
?g???X@)?}͑??1??3G?/??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?"|???Ii?@?"W@Q*?US+?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f?B??a@f?B??a@!f?B??a@      ??!       "	?v?5#@?v?5#@!?v?5#@*      ??!       2	m?kA???m?kA???!m?kA???:	A???F?"@A???F?"@!A???F?"@B      ??!       J	^??Nw?@^??Nw?@!^??Nw?@R      ??!       Z	^??Nw?@^??Nw?@!^??Nw?@b      ??!       JGPUY?"|???b qi?@?"W@y*?US+?@?"y
Ngradient_tape/model_4/time_distributed_70/conv1d_29/conv1d/Conv2DBackpropInputConv2DBackpropInput??8?J???!??8?J???0"{
Ogradient_tape/model_4/time_distributed_60/conv1d_24/conv1d/Conv2DBackpropFilterConv2DBackpropFilter@|}6???!??x???0"-
IteratorGetNext/_1_Sendx7K:.???!??mL??"{
Ogradient_tape/model_4/time_distributed_62/conv1d_25/conv1d/Conv2DBackpropFilterConv2DBackpropFilter ?0)E??!}&ˉ??0"H
,model_4/time_distributed_60/conv1d_24/conv1dConv2Du?l???!??t?-??"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??
R?פ?!?F\????0"{
Ogradient_tape/model_4/time_distributed_64/conv1d_26/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?i????!?iI?O
??0"y
Ngradient_tape/model_4/time_distributed_68/conv1d_28/conv1d/Conv2DBackpropInputConv2DBackpropInput???????!?}ER:??0"H
,model_4/time_distributed_70/conv1d_29/conv1dConv2D??????!???[??"H
,model_4/time_distributed_68/conv1d_28/conv1dConv2D?Z4M?d??!?:V)?g??Q      Y@Y?	g??@a???O8X@q??$q[?H@y	xڳ/??"?
both?Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?49.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 