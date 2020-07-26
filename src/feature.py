import collections
import multiprocessing
import os
import linecache
import threading

import tensorflow as tf

from flag_center import FLAGS


class Example(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, x_context, x_response):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          x_context: 对话上下文
          x_response: 对话下一句
        """
        self.guid = guid
        self.x_context = x_context
        self.x_response = x_response


class Features(object):
    """
    A single set of features of data.
    """

    def __init__(self,
                 x_context,
                 x_response,
                 is_real_example=True):
        """Constructs a InputExample.
           Args:
             x_context: 对话上下文
             x_response: 对话下一句
             is_real_example: 是不是使用实数
        """
        self.x_context = x_context
        self.x_response = x_response
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        lines = DataProcessor._read_raw_feature(os.path.join(data_dir, "train.txt"))
        return DataProcessor._create_example(lines=lines)

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        lines = DataProcessor._read_raw_feature(os.path.join(data_dir, "dev.txt"))
        return DataProcessor._create_example(lines=lines)

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        lines = DataProcessor._read_raw_feature(os.path.join(data_dir, "test.txt"))
        return DataProcessor._create_example(lines=lines)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1]

    @classmethod
    def _read_raw_feature(cls, input_file):
        """Reads a tab separated value file."""
        lines = linecache.getlines(filename=input_file)
        raw_features = list()
        for line in lines:
            line = line.strip()
            raw_features.append(line)
        return raw_features

    @classmethod
    def _create_example(cls, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(len(lines)):
            if i % 10000 == 0:
                tf.logging.info('create example %d' % (i))
            x_line = lines[i]
            x_line_items = x_line.split('\t')
            x_context = x_line_items[0]
            x_response = x_line_items[1]
            if int(len(x_line_items)) != 3:
                continue
            if int(float(x_line_items[2])) == 0: # 删除随机采样的负样本，在训练poly-encoder 和 bi-encoder 的时候
                continue
            examples.append(
                Example(guid='guid' + str(i), x_context=x_context, x_response=x_response)
            )
        return examples


def convert_single_example(ex_index, example, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    x_context = tokenizer.tokenize(text=example.x_context)
    x_response = tokenizer.tokenize(text=example.x_response)
    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    x_context_ids = [cls_id] + tokenizer.convert_tokens_to_ids(x_context)[-FLAGS.context_length:] + [sep_id]
    x_response_ids = [cls_id] + tokenizer.convert_tokens_to_ids(x_response)[:FLAGS.candidate_length] + [sep_id]


    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("x_context_ids: %s" % " ".join([str(x) for x in x_context_ids]))
        tf.logging.info("x_response_ids: %s" % " ".join([str(x) for x in x_response_ids]))

    feature = Features(x_context=x_context_ids, x_response=x_response_ids, is_real_example=False)
    return feature


def file_based_convert_examples_to_features(examples, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 100 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["x_context"] = create_int_feature(feature.x_context)
        features["x_response"] = create_int_feature(feature.x_response)
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    writer.close()


def file_based_input_fn_builder(input_file, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "x_context": tf.VarLenFeature(tf.int64),
        "x_response": tf.VarLenFeature(tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        #L = 500
        for name in list(name_to_features.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
                t = tf.sparse_tensor_to_dense(t)
            example[name] = t #[-L:]

        return example["x_context"], example["x_response"]

    def input_fn(params):

        """The actual input function."""
        batch_size = params["train_batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100*batch_size)
        d = d.map(
            map_func=lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=16
        )

        def pad_batch(dataset_tmp, batch_size, drop_remainder):

            padded_shapes = (
                ([None], [None])
            )
            padding_values = (
                (0, 0)
            )

            dataset_tmp = dataset_tmp.padded_batch(batch_size=batch_size,
                                                   padded_shapes=padded_shapes,
                                                   padding_values=padding_values,
                                                   drop_remainder=drop_remainder)
            return dataset_tmp
        d = pad_batch(dataset_tmp=d, batch_size=batch_size, drop_remainder=drop_remainder)
        d.prefetch(1)
        return d

    return input_fn


class FeatureThread(multiprocessing.Process):

    def __init__(self, examples, tokenizer, output_file="./dat/"):

        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.output_file = output_file

    def run(self):
        """线程内容"""
        file_based_convert_examples_to_features(examples=self.examples,
                                                tokenizer=self.tokenizer,
                                                output_file=self.output_file)

    @classmethod
    def split_task(cls, num_thrd, examples, out_dir, mode):
        mode_name = "test"
        if mode == tf.estimator.ModeKeys.TRAIN:
            mode_name = "train"
        task_params = list()
        num_examples = len(examples)
        for i in range(0, num_examples, int(len(examples) / num_thrd)):
            ele = [examples[i: i + int(len(examples) / num_thrd)],
                   os.path.join(out_dir, mode_name + "_" + str(i) + '.tfrecord')]
            task_params.append(ele)
        return task_params