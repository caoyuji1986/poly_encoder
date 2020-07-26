import tensorflow as tf
import linecache
import os

import tokenization
from flag_center import FLAGS,flags
from model import PolyEncoderConfig, PolyEncoder, BiEncoder, BiEncoderConfig

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
    for context, response_list in samples[:1000]:
        x_context = tokenizer.tokenize(text=context)
        c_token_ele = [cls_id] + tokenizer.convert_tokens_to_ids(tokens=x_context)[-(context_length-2):] + [sep_id]
        c_tokens_ret = [c_token_ele for _ in response_list]

        x_response = [tokenizer.tokenize(text=response_ele) for response_ele in response_list]
        rl_tokens = [
            [cls_id] + tokenizer.convert_tokens_to_ids(tokens=ele)[:(candidate_length-2)] + [sep_id]
            for ele in x_response
        ]
        rl_tokens_ret = []
        for i in range(len(rl_tokens)):
            remain_len = candidate_length  - len(rl_tokens[i])
            rl_tokens_ret.append(rl_tokens[i] + [0 for ele in range(remain_len)])
        features.append([c_tokens_ret, rl_tokens_ret])

    return features


def create_model(model_config_file, k, model_type='poly-encoder'):

    x_context_tensor = tf.placeholder(dtype=tf.int32, shape=[None, None])
    x_response_tensor = tf.placeholder(dtype=tf.int32, shape=[None, None])
    if model_type == 'bi-encoder':
        config = BiEncoderConfig.from_json_file(json_file=model_config_file)
        inst = BiEncoder(config=config, mode=tf.estimator.ModeKeys.EVAL)
    else:
        config = PolyEncoderConfig.from_json_file(json_file=model_config_file)
        inst = PolyEncoder(config=config, mode=tf.estimator.ModeKeys.EVAL)
    context_emb_norm, candidate_emb_norm = \
        inst.create_model(x_context=x_context_tensor, x_response=x_response_tensor)
    dist = inst.calculate_distance(context_emb_norm=context_emb_norm, candidate_emb_norm=candidate_emb_norm)
    value, idx = tf.nn.top_k(input=dist, k=k, sorted=True)
    return (idx, value), x_context_tensor, x_response_tensor


def main(unused):

    k = FLAGS.recall_k
    model_config_file = FLAGS.model_config
    data_dir = os.path.join(FLAGS.data_dir, 'test.txt')
    vocab_file = FLAGS.vocab_file
    context_length = FLAGS.context_length
    candidate_length = FLAGS.candidate_length
    idx, x_context_tensor, x_response_tensor = create_model(model_config_file=model_config_file, k=k, model_type=FLAGS.model_type)
    samples = load_eval_samples(file_name=data_dir)
    features = convert_samples_to_features(samples=samples, vocab_file=vocab_file,
                                           context_length=context_length, candidate_length=candidate_length)
    sess = tf.Session()
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    saver.restore(sess=sess, save_path=model_file)
    #sess.run(tf.global_variables_initializer())
    hit_num = 0
    for feature in features:
        # batch_size x 5
        recall_k_in_10, recall_k_in_10_val = sess.run(fetches=idx, feed_dict={
            x_context_tensor: feature[0],
            x_response_tensor: feature[1]
        })
        if 0 in recall_k_in_10:
            hit_num += 1
            print(hit_num)
    print("Result-------------------------------------------------------")
    print("recall@%d/10: %f" % (k, float(hit_num)/float(len(features))))
    print("-------------------------------------------------------------")


if __name__ == '''__main__''':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()