?	?`?d?f@?`?d?f@!?`?d?f@	:8a??$??:8a??$??!:8a??$??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?`?d?f@????l?c@15???"@A_?iN^d??I?A|`Ǉ'@Y1@?	?@*	;?O???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator(?r?w@!]?x|?X@)(?r?w@1]?x|?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?\???@!L2??X@)??h?????1>T[?_H??:Preprocessing2F
Iterator::Modelb??A?@!      Y@)??[v?x?1(.A??T??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismx??1!?@!????*?X@)??K?ut?1<?^?y*??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9:8a??$??I??=?EGW@Q???t?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????l?c@????l?c@!????l?c@      ??!       "	5???"@5???"@!5???"@*      ??!       2	_?iN^d??_?iN^d??!_?iN^d??:	?A|`Ǉ'@?A|`Ǉ'@!?A|`Ǉ'@B      ??!       J	1@?	?@1@?	?@!1@?	?@R      ??!       Z	1@?	?@1@?	?@!1@?	?@b      ??!       JGPUY:8a??$??b q??=?EGW@y???t?@?"y
Ngradient_tape/model_1/time_distributed_25/conv1d_11/conv1d/Conv2DBackpropInputConv2DBackpropInput?p?!&??!?p?!&??0"z
Ngradient_tape/model_1/time_distributed_15/conv1d_6/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?Dz?+??!?Z??????0"z
Ngradient_tape/model_1/time_distributed_17/conv1d_7/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?M?? ???!^?b&????0"G
+model_1/time_distributed_15/conv1d_6/conv1dConv2DF?4?i???!??I????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?n?(Bz??!v????K??0"-
IteratorGetNext/_1_Sendi?d????!?;i????"z
Ngradient_tape/model_1/time_distributed_19/conv1d_8/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?|??`??!D?, ?6??0"y
Ngradient_tape/model_1/time_distributed_23/conv1d_10/conv1d/Conv2DBackpropInputConv2DBackpropInput`"cA(ݡ?!?0Y(?r??0"H
,model_1/time_distributed_25/conv1d_11/conv1dConv2D"	?i??!ss}I????"H
,model_1/time_distributed_23/conv1d_10/conv1dConv2D????
Ϡ?!???????Q      Y@Y??N??N@a?;?;W@q?^Ej??3@yG.??բ?"?
both?Your program is POTENTIALLY input-bound because 86.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?19.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 