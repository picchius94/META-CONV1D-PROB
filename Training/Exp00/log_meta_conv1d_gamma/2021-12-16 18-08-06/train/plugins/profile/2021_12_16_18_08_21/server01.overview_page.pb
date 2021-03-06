?	???Aާp@???Aާp@!???Aާp@	???Q@???Q@!???Q@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???Aާp@nm?y?%n@1?Ց##@A/4?i????Ia?xwd| @Yw?$z@*	n???[?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generatorй??Ҭ @!^???X@)й??Ҭ @1^???X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap;oc?#? @!?+??p?X@)?j{???1t??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Hh˹? @!??9??X@)Rf`X??1???/????:Preprocessing2F
Iterator::ModelL?{)<? @!      Y@)am???|?1Ҍ7???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???Q@I?l3L?iW@Q?]ዄ?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	nm?y?%n@nm?y?%n@!nm?y?%n@      ??!       "	?Ց##@?Ց##@!?Ց##@*      ??!       2	/4?i????/4?i????!/4?i????:	a?xwd| @a?xwd| @!a?xwd| @B      ??!       J	w?$z@w?$z@!w?$z@R      ??!       Z	w?$z@w?$z@!w?$z@b      ??!       JGPUY???Q@b q?l3L?iW@y?]ዄ?@?"y
Ngradient_tape/model_3/time_distributed_55/conv1d_23/conv1d/Conv2DBackpropInputConv2DBackpropInput???N???!???N???0"{
Ogradient_tape/model_3/time_distributed_45/conv1d_18/conv1d/Conv2DBackpropFilterConv2DBackpropFilter~?CN?ٱ?!&|???d??0"-
IteratorGetNext/_1_Send??6]Y??! ??@?:??"{
Ogradient_tape/model_3/time_distributed_47/conv1d_19/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?f5a?I??!n,w,?&??0"H
,model_3/time_distributed_45/conv1d_18/conv1dConv2D?r?%B??!???????"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?Ư?0??!?{???t??0"{
Ogradient_tape/model_3/time_distributed_49/conv1d_20/conv1d/Conv2DBackpropFilterConv2DBackpropFilter^a}????!?'??c???0"y
Ngradient_tape/model_3/time_distributed_53/conv1d_22/conv1d/Conv2DBackpropInputConv2DBackpropInput??E?????!??Sk????0"H
,model_3/time_distributed_55/conv1d_23/conv1dConv2Dg?????!=⒝?	??"H
,model_3/time_distributed_53/conv1d_22/conv1dConv2D?*^?勠?!??^[2??Q      Y@Y??N??N@a?;?;W@q??غ??I@yL???@??"?
both?Your program is POTENTIALLY input-bound because 90.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?51.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 