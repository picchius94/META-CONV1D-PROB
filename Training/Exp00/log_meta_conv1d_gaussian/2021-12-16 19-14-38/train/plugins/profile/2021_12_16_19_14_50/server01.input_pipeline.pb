	?|?X%]d@?|?X%]d@!?|?X%]d@	j?????@j?????@!j?????@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?|?X%]d@öE?q`@1??{?9#@AB	3m????I???V?+@Y73??pj@*	? ?r???@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator??e	"@!??3?X@)??e	"@1??3?X@:Preprocessing2F
Iterator::Model???/J"@!      Y@)?7? ?x?1??!J?6??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?3?/."@!?w?V??X@)?+???t?1H???x??:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMapK?b??
"@!??T#?X@):?Y?Xh?1??>?
٠?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 80.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9j?????@I?۾??[V@Q(???D?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	öE?q`@öE?q`@!öE?q`@      ??!       "	??{?9#@??{?9#@!??{?9#@*      ??!       2	B	3m????B	3m????!B	3m????:	???V?+@???V?+@!???V?+@B      ??!       J	73??pj@73??pj@!73??pj@R      ??!       Z	73??pj@73??pj@!73??pj@b      ??!       JGPUYj?????@b q?۾??[V@y(???D?@