	??PM?<?@??PM?<?@!??PM?<?@	???:?T@???:?T@!???:?T@"w
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
	?ʾ+?;g@?ʾ+?;g@!?ʾ+?;g@      ??!       "	]j?~??"@]j?~??"@!]j?~??"@*      ??!       2	?????f???????f??!?????f??:	?+??2?%@?+??2?%@!?+??2?%@B      ??!       J	;?s?Q?@;?s?Q?@!;?s?Q?@R      ??!       Z	;?s?Q?@;?s?Q?@!;?s?Q?@b      ??!       JGPUY???:?T@b qן[???2@y???J??