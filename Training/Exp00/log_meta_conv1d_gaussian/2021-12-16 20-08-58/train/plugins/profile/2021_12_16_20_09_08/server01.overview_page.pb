?	mW??e?g@mW??e?g@!mW??e?g@	8VJJ%+@8VJJ%+@!8VJJ%+@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6mW??e?g@?B?ʠ3a@1p>u?R?"@Ah?>????I?? @?:0@YSy=??9@*	?$??h?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?֥F?;@!d??q?X@)?֥F?;@1d??q?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?3?;@!?A?J?X@)??t?i???1???Ʈ?:Preprocessing2F
Iterator::Model? ?}?
;@!      Y@)? ??zy?1??(_^0??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismbHN&n	;@!t???X@)L4H?S?u?12??<#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t72.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.98VJJ%+@I&???\T@Q1Q?͓?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?B?ʠ3a@?B?ʠ3a@!?B?ʠ3a@      ??!       "	p>u?R?"@p>u?R?"@!p>u?R?"@*      ??!       2	h?>????h?>????!h?>????:	?? @?:0@?? @?:0@!?? @?:0@B      ??!       J	Sy=??9@Sy=??9@!Sy=??9@R      ??!       Z	Sy=??9@Sy=??9@!Sy=??9@b      ??!       JGPUY8VJJ%+@b q&???\T@y1Q?͓?@?"y
Ngradient_tape/model_2/time_distributed_40/conv1d_17/conv1d/Conv2DBackpropInputConv2DBackpropInput?cFqX???!?cFqX???0"{
Ogradient_tape/model_2/time_distributed_30/conv1d_12/conv1d/Conv2DBackpropFilterConv2DBackpropFilteroGϥ?<??!?Պ????0"-
IteratorGetNext/_1_Send??>??!&E??i???"{
Ogradient_tape/model_2/time_distributed_32/conv1d_13/conv1d/Conv2DBackpropFilterConv2DBackpropFilter???i׸??!HA???f??0"H
,model_2/time_distributed_30/conv1d_12/conv1dConv2D?;??o5??!¨,?}??"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropqJ??n??!?J????0"{
Ogradient_tape/model_2/time_distributed_34/conv1d_14/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??pzV??!?EN?????0"y
Ngradient_tape/model_2/time_distributed_38/conv1d_16/conv1d/Conv2DBackpropInputConv2DBackpropInput82?ӻ???!5,?UR-??0"H
,model_2/time_distributed_40/conv1d_17/conv1dConv2D+5??"??!?r???Q??"H
,model_2/time_distributed_38/conv1d_16/conv1dConv2DE??I???!?,?,b??Q      Y@Y?T?x?r
@a[=;n,X@q?0?965@y??M^?0??"?
both?Your program is MODERATELY input-bound because 13.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t72.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?21.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 