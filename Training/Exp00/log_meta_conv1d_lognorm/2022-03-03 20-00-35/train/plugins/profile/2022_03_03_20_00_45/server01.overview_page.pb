?	?X??+Kf@?X??+Kf@!?X??+Kf@	=]!?bH@=]!?bH@!=]!?bH@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?X??+Kf@s,???a@1` ??#@A????o??I?_?L?0@Y.?l?I"@*	??v???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorXWj1 %@!?%?b6?X@)XWj1 %@1?%?b6?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap? ?!?&%@!??)S??X@)??&3?V??1>`????:Preprocessing2F
Iterator::Model??V%?-%@!      Y@)J??	?y{?1????7??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`?5?!*%@!? ??X@)}?E?z?1?xW?&???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t80.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9=]!?bH@I??2eV@QV21+|d@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	s,???a@s,???a@!s,???a@      ??!       "	` ??#@` ??#@!` ??#@*      ??!       2	????o??????o??!????o??:	?_?L?0@?_?L?0@!?_?L?0@B      ??!       J	.?l?I"@.?l?I"@!.?l?I"@R      ??!       Z	.?l?I"@.?l?I"@!.?l?I"@b      ??!       JGPUY=]!?bH@b q??2eV@yV21+|d@?"y
Ngradient_tape/model_3/time_distributed_55/conv1d_23/conv1d/Conv2DBackpropInputConv2DBackpropInput?:??~??!?:??~??0"{
Ogradient_tape/model_3/time_distributed_45/conv1d_18/conv1d/Conv2DBackpropFilterConv2DBackpropFilter????@??!??_??0"H
,model_3/time_distributed_45/conv1d_18/conv1dConv2D':??fS??!de?/?s??"{
Ogradient_tape/model_3/time_distributed_47/conv1d_19/conv1d/Conv2DBackpropFilterConv2DBackpropFilterr!*.J??!??H[=???0"-
IteratorGetNext/_1_Sendy?'???!??W"???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?ͻ??
??!z?O?wb??0"{
Ogradient_tape/model_3/time_distributed_49/conv1d_20/conv1d/Conv2DBackpropFilterConv2DBackpropFilter'DN?+??!??v????0"y
Ngradient_tape/model_3/time_distributed_53/conv1d_22/conv1d/Conv2DBackpropInputConv2DBackpropInput???$?z??!????H???0"H
,model_3/time_distributed_55/conv1d_23/conv1dConv2DT??K??!&?4r???"H
,model_3/time_distributed_53/conv1d_22/conv1dConv2D????`??!{Hm???Q      Y@Y>???>@a>???>X@q?F/?x?+@y?? ?????"?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t80.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?13.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 