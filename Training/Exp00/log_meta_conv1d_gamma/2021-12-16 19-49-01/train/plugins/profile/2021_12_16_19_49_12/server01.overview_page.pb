?	?_=??Bh@?_=??Bh@!?_=??Bh@	S?@7c???S?@7c???!S?@7c???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?_=??Bh@?`ob?6e@1??k???"@ACT???f??I?6???Z*@Y{C???*	7?A`eH?@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator|?&?@!?bu?X@)|?&?@1?bu?X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap|??l;?@!?n????X@)k??P????1l?q?N%??:Preprocessing2F
Iterator::Model?u?;O?@!      Y@)??U????1?,??!???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism 
fL?@!?T?z?X@)??L???16E??޺??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9S?@7c???IO#?(?W@QB?U	2]@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?`ob?6e@?`ob?6e@!?`ob?6e@      ??!       "	??k???"@??k???"@!??k???"@*      ??!       2	CT???f??CT???f??!CT???f??:	?6???Z*@?6???Z*@!?6???Z*@B      ??!       J	{C???{C???!{C???R      ??!       Z	{C???{C???!{C???b      ??!       JGPUYS?@7c???b qO#?(?W@yB?U	2]@?"y
Ngradient_tape/model_4/time_distributed_70/conv1d_29/conv1d/Conv2DBackpropInputConv2DBackpropInput΂M??!΂M??0"{
Ogradient_tape/model_4/time_distributed_60/conv1d_24/conv1d/Conv2DBackpropFilterConv2DBackpropFilterKcF?8???!??dH(???0"{
Ogradient_tape/model_4/time_distributed_62/conv1d_25/conv1d/Conv2DBackpropFilterConv2DBackpropFilterS$?r2???!?!?t???0"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??Y^M˥?!4MQ????0"H
,model_4/time_distributed_60/conv1d_24/conv1dConv2D!]$?(???!?ؕ3?Q??"-
IteratorGetNext/_1_Sendm?f?t???!f???W??"{
Ogradient_tape/model_4/time_distributed_64/conv1d_26/conv1d/Conv2DBackpropFilterConv2DBackpropFilterU???U??!?Z?
O??0"y
Ngradient_tape/model_4/time_distributed_68/conv1d_28/conv1d/Conv2DBackpropInputConv2DBackpropInput?X??]ġ?!!g????0"H
,model_4/time_distributed_70/conv1d_29/conv1dConv2D?1?a?j??!aQS????"H
,model_4/time_distributed_68/conv1d_28/conv1dConv2D?Ϟ?ࡠ?!T??g#???Q      Y@Y??N??N@a?;?;W@qpTS{"#E@y???w X??"?
both?Your program is POTENTIALLY input-bound because 87.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?6.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?42.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 