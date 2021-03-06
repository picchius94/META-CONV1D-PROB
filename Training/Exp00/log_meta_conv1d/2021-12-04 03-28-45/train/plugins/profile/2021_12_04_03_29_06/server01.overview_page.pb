?	??PM?<?@??PM?<?@!??PM?<?@	???:?T@???:?T@!???:?T@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??PM?<?@?ʾ+?;g@1]j?~??"@A?????f??I?+??2?%@Y;?s?Q?@*	???Hn)A2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator?P??Q
?@!?????X@)?P??Q
?@1?????X@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap?? 4j
?@!G????X@)??	h"l??1?<:q,rW?:Preprocessing2F
Iterator::Model?h>?
?@!      Y@)??'?ځ?1??$Q?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???cu
?@!#?????X@)???uR_v?1????WzE?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 80.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t17.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???:?T@Iן[???2@Q???J??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ʾ+?;g@?ʾ+?;g@!?ʾ+?;g@      ??!       "	]j?~??"@]j?~??"@!]j?~??"@*      ??!       2	?????f???????f??!?????f??:	?+??2?%@?+??2?%@!?+??2?%@B      ??!       J	;?s?Q?@;?s?Q?@!;?s?Q?@R      ??!       Z	;?s?Q?@;?s?Q?@!;?s?Q?@b      ??!       JGPUY???:?T@b qן[???2@y???J???"y
Ngradient_tape/model_2/time_distributed_40/conv1d_17/conv1d/Conv2DBackpropInputConv2DBackpropInput??s?????!??s?????0"-
IteratorGetNext/_1_SendGo??Cݴ?!??0d???"{
Ogradient_tape/model_2/time_distributed_30/conv1d_12/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?k?=??!u?Ky??0"{
Ogradient_tape/model_2/time_distributed_32/conv1d_13/conv1d/Conv2DBackpropFilterConv2DBackpropFilter7E?'?!??07??0"H
,model_2/time_distributed_30/conv1d_12/conv1dConv2D???h???!w/ED???"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop??????!??BCl??0"{
Ogradient_tape/model_2/time_distributed_34/conv1d_14/conv1d/Conv2DBackpropFilterConv2DBackpropFilter?,??(??!L??EX???0"y
Ngradient_tape/model_2/time_distributed_38/conv1d_16/conv1d/Conv2DBackpropInputConv2DBackpropInput䫂1????!????????0"H
,model_2/time_distributed_40/conv1d_17/conv1dConv2D[?tI?>??!Ӌ	???"H
,model_2/time_distributed_38/conv1d_16/conv1dConv2D; 	<???!???Ȓ??Q      Y@Y?Oq??@a?h?>?W@q????? @y?
???}??"?

host?Your program is HIGHLY input-bound because 80.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t17.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 