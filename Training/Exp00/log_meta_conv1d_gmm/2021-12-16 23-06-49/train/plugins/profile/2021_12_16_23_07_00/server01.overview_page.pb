?	2?	???c@2?	???c@!2?	???c@	-?I?ט@-?I?ט@!-?I?ט@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails62?	???c@?o&??_@1????}?"@A??x"????I?Z?7ڹ+@Y?????@*	?ʡE??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?<֌d"@!Y?T?\?X@)?<֌d"@1Y?T?\?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??+Hs"@!o??J?X@)??X?v??1???^???:Preprocessing2F
Iterator::Modele?uz"@!      Y@)????1??r?N??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism3#?v"@!?TcQ??X@)???1?y?1??EwɅ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 80.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9.?I?ט@I??{B?FV@QR??$??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?o&??_@?o&??_@!?o&??_@      ??!       "	????}?"@????}?"@!????}?"@*      ??!       2	??x"??????x"????!??x"????:	?Z?7ڹ+@?Z?7ڹ+@!?Z?7ڹ+@B      ??!       J	?????@?????@!?????@R      ??!       Z	?????@?????@!?????@b      ??!       JGPUY.?I?ט@b q??{B?FV@yR??$??@?"y
Ngradient_tape/model_2/time_distributed_40/conv1d_17/conv1d/Conv2DBackpropInputConv2DBackpropInput?DlP???!?DlP???0"{
Ogradient_tape/model_2/time_distributed_30/conv1d_12/conv1d/Conv2DBackpropFilterConv2DBackpropFilter|Î,)???!gi̼p??0"-
IteratorGetNext/_1_SendJ???۩?!?ִ????"{
Ogradient_tape/model_2/time_distributed_32/conv1d_13/conv1d/Conv2DBackpropFilterConv2DBackpropFilterBe}???!*????0"H
,model_2/time_distributed_30/conv1d_12/conv1dConv2D>?ݞ?`??!r}?z????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?#????!???]?T??0"{
Ogradient_tape/model_2/time_distributed_34/conv1d_14/conv1d/Conv2DBackpropFilterConv2DBackpropFilterD"???!.F{????0"y
Ngradient_tape/model_2/time_distributed_38/conv1d_16/conv1d/Conv2DBackpropInputConv2DBackpropInput??|Qx???!??M?????0"H
,model_2/time_distributed_40/conv1d_17/conv1dConv2D{???=??!Q?Ё}???"H
,model_2/time_distributed_38/conv1d_16/conv1dConv2D???:???!?Ƭޤ??Q      Y@Y?	g??@a???O8X@qj???G?.@y??m0H??"?
both?Your program is POTENTIALLY input-bound because 80.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?15.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 