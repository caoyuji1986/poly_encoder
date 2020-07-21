import os,time
import tokenization
import tensorflow as tf
from feature import DataProcessor, file_based_convert_examples_to_features, file_based_input_fn_builder


def test_DataProcessor():

    data_processor = DataProcessor()
    examples = data_processor.get_train_examples('./dat/')

    def str_example(example):

        s = ""
        s += example.x_context + "\t" + example.x_response
        return s

    for example in examples:
        print(str_example(example))


def test_file_based_convert_examples_to_features():

    data_processor = DataProcessor()
    examples = data_processor.get_train_examples('./dat/')

    tokenizer = tokenization.FullTokenizer(vocab_file='../../ckpt/albert/vocab.txt', do_lower_case=True)
    cls_id = tokenizer.vocab['[CLS]']
    tf_path = os.path.join('./dat/', 'train.tfrecord')
    file_based_convert_examples_to_features(examples=examples, tokenizer=tokenizer, output_file=tf_path)


def test_file_based_input_fn_builder():

    tf_path = os.path.join('./dat/', 'train.tfrecord')
    input_fn = file_based_input_fn_builder(input_file=tf_path, is_training=False, drop_remainder=False)
    params = dict()
    params["train_batch_size"] = 2
    d = input_fn(params=params)
    iter = tf.data.Iterator.from_structure(d.output_types, d.output_shapes)
    xs = iter.get_next()
    test_init_op = iter.make_initializer(d)

    with tf.Session() as sess:
        sess.run(test_init_op)

        while True:
            ret_x_context, ret_x_response = sess.run([xs[0], xs[1]])
            print("-------------------------------------")
            for ele in ret_x_context:
                print(','.join([str(ele_i) for ele_i in ele]))
            for ele in ret_x_response:
                print(','.join([str(ele_i) for ele_i in ele]))
            time.sleep(1)
            break

if __name__=='''__main__''':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    #test_DataProcessor()
    test_file_based_input_fn_builder()
    #test_file_based_input_fn_builder()