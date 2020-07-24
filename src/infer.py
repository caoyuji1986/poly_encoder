import tensorflow as tf
import linecache
import os

import tokenization
from flag_center import FLAGS,flags
from model import PolyEncoderConfig, PolyEncoder

flags.DEFINE_integer(name='recall_k', default=3, help='k值')


def load_eval_samples(file_name):

    samples = list()
    lines = linecache.getlines(filename=file_name)
    for line in lines:
        line = line.strip()
        items = line.split("\t")
        context = items[0]
        response_list = items[1:]

        samples.append([context, response_list])
    return samples


def convert_samples_to_features(samples, vocab_file, context_length, candidate_length):

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    features = list()
    for context, response_list in samples:
        c_tokens = [[cls_id] + tokenizer.tokenize(text=context)[-context_length:] + [sep_id] for _ in response_list]
        rl_tokens = [[cls_id] + tokenizer.tokenize(ele)[:candidate_length] + [sep_id] for ele in response_list]
        features.append([c_tokens, rl_tokens])
    return features


def crerate_model(model_config_file, k):

    x_context_tensor = tf.placeholder(dtype=tf.int32, shape=[None, None])
    x_response_tensor = tf.placeholder(dtype=tf.int32, shape=[None, None])
    config = PolyEncoderConfig.from_json_file(json_file=model_config_file)
    inst = PolyEncoder(config=config, mode=tf.estimator.ModeKeys.EVAL)
    context_emb_norm, x_response_hidden_norm = \
        inst.create_model(x_context=x_context_tensor, x_response=x_response_tensor)
    dist = inst.calculate_distance(context_emb_norm=context_emb_norm, x_response_hidden_norm=x_response_hidden_norm)
    value, idx = tf.nn.top_k(input=dist, k=k, sorted=True)
    return idx, x_context_tensor, x_response_tensor


def main(unused):

    k = FLAGS.recall_k
    model_config_file = FLAGS.model_config
    data_dir = os.path.join(FLAGS.data_dir, 'test.txt')
    vocab_file = FLAGS.vocab_file
    context_length = FLAGS.context_length
    candidate_length = FLAGS.candidate_length
    idx, x_context_tensor, x_response_tensor = crerate_model(model_config_file=model_config_file, k=k)
    samples = load_eval_samples(file_name=data_dir)
    features = convert_samples_to_features(samples=samples, vocab_file=vocab_file,
                                           context_length=context_length, candidate_length=candidate_length)
    sess = tf.Session()
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    saver.restore(sess=sess, save_path=model_file)
    hit_num = 0
    for feature in features:
        # batch_size x 5
        recall_k_in_10 = sess.run(fetches=idx, feed_dict={
            x_context_tensor: feature[0],
            x_response_tensor: feature[1]
        })
        for ele in recall_k_in_10:
            if 1 in ele:
                hit_num += 1
    print("Result-------------------------------------------------------")
    print("recall@%d/10: %f" % (k, float(hit_num)/float(len(features))))
    print("-------------------------------------------------------------")


if __name__ == '''__main__''':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()