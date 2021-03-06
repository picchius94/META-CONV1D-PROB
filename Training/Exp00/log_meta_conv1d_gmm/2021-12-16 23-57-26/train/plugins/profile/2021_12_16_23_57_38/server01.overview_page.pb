?	y]?`?<e@y]?`?<e@!y]?`?<e@	i*???@i*???@!i*???@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6y]?`?<e@V?y??`@1c??K??"@A?CԷ???I#?tuǺ0@YiT?d?!@*	??~jh?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorDԷ??&@!?^????X@)DԷ??&@1?^????X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapQ?|?&@!V5???X@)?W?????1gs??V???:Preprocessing2F
Iterator::ModelU????&@!      Y@).??T???1??{?2??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismU?]?&@!|!_??X@)A?)V?|?18+I??U??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t79.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9i*???@In????RV@Q??-n?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V?y??`@V?y??`@!V?y??`@      ??!       "	c??K??"@c??K??"@!c??K??"@*      ??!       2	?CԷ????CԷ???!?CԷ???:	#?tuǺ0@#?tuǺ0@!#?tuǺ0@B      ??!       J	iT?d?!@iT?d?!@!iT?d?!@R      ??!       Z	iT?d?!@iT?d?!@!iT?d?!@b      ??!       JGPUYi*???@b qn????RV@y??-n?@?"y
Ngradient_tape/model_3/time_distributed_55/conv1d_23/conv1d/Conv2DBackpropInputConv2DBackpropInput?fX?wJ??!?fX?wJ??0"{
Ogradient_tape/model_3/time_distributed_45/conv1d_18/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??0????!$?:??0"{
Ogradient_tape/model_3/time_distributed_47/conv1d_19/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?U?~y???!?O?l?T??0"H
,model_3/time_distributed_45/conv1d_18/conv1dConv2D????춥?!?(?)???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropk,?Bq??!7
?R???0"{
Ogradient_tape/model_3/time_distributed_49/conv1d_20/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?U??L???!??Ҙ????0"-
IteratorGetNext/_1_Send?XJD{??!@\1??"y
Ngradient_tape/model_3/time_distributed_53/conv1d_22/conv1d/Conv2DBackpropInputConv2DBackpropInputd?Y
?ޡ?!?y???l??0"H
,model_3/time_distributed_55/conv1d_23/conv1dConv2Dt|?[???!`i?[???"H
,model_3/time_distributed_53/conv1d_22/conv1dConv2DsF?|???!.r??J???Q      Y@Y?	g??@a???O8X@q>{Nqd?@yL=????"?
both?Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t79.3 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 