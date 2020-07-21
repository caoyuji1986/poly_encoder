import collections

import tensorflow as tf

values = [1,2,3,4,5,6,7,8,9]
values1 = [1,2,3,4,5,6,7,8]
f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values1)))
f_list1 = tf.train.FeatureList(feature=[f, f1])

input_file = './dat/out.txt'
writer = tf.python_io.TFRecordWriter(input_file)

features = collections.OrderedDict()
features["x_context"] = f
feature_lists = collections.OrderedDict()
feature_lists["x_response"] = f_list1

tf_example = tf.train.SequenceExample(context=tf.train.Features(feature=features),
                                      feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
writer.write(tf_example.SerializeToString())


values = [1,2,3,4,5,6,7,8,9]
values1 = [1,2,3,4,5,6]
f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values1)))
f_list2 = tf.train.FeatureList(feature=[f, f1])

features = collections.OrderedDict()
features["x_context"] = f
feature_lists = collections.OrderedDict()
feature_lists["x_response"] = f_list2

tf_example = tf.train.SequenceExample(context=tf.train.Features(feature=features),
                                      feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
writer.write(tf_example.SerializeToString())



values = [1,2,3,4,5]
values1 = [1,2,3,4,5,6]
f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values1)))
f_list3 = tf.train.FeatureList(feature=[f, f1])

features = collections.OrderedDict()
features["x_context"] = f
feature_lists = collections.OrderedDict()
feature_lists["x_response"] = f_list3


tf_example = tf.train.SequenceExample(context=tf.train.Features(feature=features),
                                      feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
writer.write(tf_example.SerializeToString())


writer.close()
context_features = {
    "x_context": tf.VarLenFeature(dtype=tf.int64)
}
sequence_features = {
    "x_response": tf.VarLenFeature(dtype=tf.int64)
}

def file_base_input_fn(input_file=input_file):

    d = tf.data.TFRecordDataset(input_file)


    def _decode_record(record):
        """Decodes a record to a TensorFlow example."""
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=record,
                                                   context_features=context_features,
                                                   sequence_features=sequence_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # L = 500
        x_context = tf.to_int32(context_parsed['x_context'])
        x_response = tf.to_int32(sequence_parsed['x_response'])


        return tf.sparse_tensor_to_dense(x_context), tf.sparse_tensor_to_dense(x_response)

    d = d.map(map_func=lambda record: _decode_record(record), num_parallel_calls=16)

    def pad_batch(dataset_tmp, batch_size, drop_remainder):
        padded_shapes = (
            ([None], [None, None])
        )
        padding_values = (
            (0, 0)
        )
        dataset_tmp = dataset_tmp.padded_batch(batch_size=batch_size,
                                               padded_shapes=padded_shapes,
                                               padding_values=padding_values,
                                               drop_remainder=drop_remainder)
        return dataset_tmp
    d = pad_batch(d, 3, False)
    return d



d = file_base_input_fn()
iter = d.make_initializable_iterator()

one_element = iter.get_next()
with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(one_element))

    print(sess.run(one_element))