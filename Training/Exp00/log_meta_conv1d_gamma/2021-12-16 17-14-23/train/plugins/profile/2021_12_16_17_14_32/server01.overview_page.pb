?	?t???Ze@?t???Ze@!?t???Ze@	?+n?!?@?+n?!?@!?+n?!?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?t???Ze@ew?hea@1;ŪA?{"@A??b?????IB???8?&@YΨ?*??%@*?x?&?w?@)      ?=2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?????'@!bd????X@)?????'@1bd????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????(@!??N+??X@)???]????1Cʵ?????:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap????k?'@!?׉???X@)?S;?Ԗ??1R??U&???:Preprocessing2F
Iterator::Model><K?(@!      Y@)1{?v???1?5?R˱?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t81.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?+n?!?@I?,V@Q??q??@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ew?hea@ew?hea@!ew?hea@      ??!       "	;ŪA?{"@;ŪA?{"@!;ŪA?{"@*      ??!       2	??b???????b?????!??b?????:	B???8?&@B???8?&@!B???8?&@B      ??!       J	Ψ?*??%@Ψ?*??%@!Ψ?*??%@R      ??!       Z	Ψ?*??%@Ψ?*??%@!Ψ?*??%@b      ??!       JGPUY?+n?!?@b q?,V@y??q??@?"y
Ngradient_tape/model_2/time_distributed_40/conv1d_17/conv1d/Conv2DBackpropInputConv2DBackpropInput{???x??!{???x??0"{
Ogradient_tape/model_2/time_distributed_30/conv1d_12/conv1d/Conv2DBackpropFilterConv2DBackpropFilteryl%??!˗??????0"{
Ogradient_tape/model_2/time_distributed_32/conv1d_13/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?|??j??!??X?k??0"H
,model_2/time_distributed_30/conv1d_12/conv1dConv2D?r?2'???!ө??????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropw*W?~ĥ?!"??j{??0"-
IteratorGetNext/_1_Send`?n{?T??!?cu???"{
Ogradient_tape/model_2/time_distributed_34/conv1d_14/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?rW???!????O\??0"y
Ngradient_tape/model_2/time_distributed_38/conv1d_16/conv1d/Conv2DBackpropInputConv2DBackpropInput?%k?X???!6	?????0"H
,model_2/time_distributed_40/conv1d_17/conv1dConv2Df?n4???!?5m0????"H
,model_2/time_distributed_38/conv1d_16/conv1dConv2D?T??[??!??=?????Q      Y@Y??N??N@a?;?;W@q|?yoW0@y??儂??"?
both?Your program is MODERATELY input-bound because 6.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t81.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 