?	bhur?+d@bhur?+d@!bhur?+d@	;y?w?@;y?w?@!;y?w?@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6bhur?+d@5Lkӻ_@1??(yu#@A}v?uŌ??I??ʅ??-@Y?'?X$@*	??K7i??@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::GeneratorJ$??(>'@!?O?V??X@)J$??(>'@1?O?V??X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?GG?E'@!??CN??X@)??(???1??[g???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismF??}?K'@!8¤?B?X@)??]?p??1X??6??:Preprocessing2F
Iterator::ModelgE?DO'@!      Y@)Vb????{?1?>??b???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t78.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9<y?w?@IS?}?U@QU????@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	5Lkӻ_@5Lkӻ_@!5Lkӻ_@      ??!       "	??(yu#@??(yu#@!??(yu#@*      ??!       2	}v?uŌ??}v?uŌ??!}v?uŌ??:	??ʅ??-@??ʅ??-@!??ʅ??-@B      ??!       J	?'?X$@?'?X$@!?'?X$@R      ??!       Z	?'?X$@?'?X$@!?'?X$@b      ??!       JGPUY<y?w?@b qS?}?U@yU????@?"y
Ngradient_tape/model_4/time_distributed_70/conv1d_29/conv1d/Conv2DBackpropInputConv2DBackpropInput?[s@{??!?[s@{??0"{
Ogradient_tape/model_4/time_distributed_60/conv1d_24/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?q??:???!??x??.??0"H
,model_4/time_distributed_60/conv1d_24/conv1dConv2D?43????!?EY?U??"{
Ogradient_tape/model_4/time_distributed_62/conv1d_25/conv1d/Conv2DBackpropFilterConv2DBackpropFilter8w^!?=??!j??0????0"-
IteratorGetNext/_1_Send|Y?D???!??ȵ???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop~ 7????!????+J??0"{
Ogradient_tape/model_4/time_distributed_64/conv1d_26/conv1d/Conv2DBackpropFilterConv2DBackpropFilterI8?	x??!%;?????0"y
Ngradient_tape/model_4/time_distributed_68/conv1d_28/conv1d/Conv2DBackpropInputConv2DBackpropInput??U5{w??!???????0"H
,model_4/time_distributed_70/conv1d_29/conv1dConv2D'??5???!???????"H
,model_4/time_distributed_68/conv1d_28/conv1dConv2D???6(b??!*w?@???Q      Y@Y>???>@a>???>X@q??3O??4@y?.?aDƣ?"?
both?Your program is MODERATELY input-bound because 6.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t78.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 