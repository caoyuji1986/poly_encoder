import tensorflow as tf
import linecache

import tokenization
from model import PolyEncoderConfig, PolyEncoder


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


def convert_samples_to_features(samples):

    tokenizer = tokenization.FullTokenizer(vocab_file='./ckpt/albert/vocab.txt', do_lower_case=True)
    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    features = list()
    for context, response_list in samples:
        c_tokens = [[cls_id] + tokenizer.tokenize(text=context)[-312:] + [sep_id] for _ in response_list]
        rl_tokens = [[cls_id] + tokenizer.tokenize(ele)[:162] + [sep_id] for ele in response_list]
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


if __name__ == '''__main__''':

    k = 5
    idx, x_context_tensor, x_response_tensor = crerate_model(model_config_file="./cfg/poly_encoder.json", k=k)
    samples = load_eval_samples(file_name="./dat/ubuntu/test.txt")
    features = convert_samples_to_features(samples=samples)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer)
    hit_num = 0
    for feature in features:
        # batch_size x 5
        recall_5_in_10 = sess.run(fetches=idx, feed_dict={
            x_context_tensor: feature[0],
            x_response_tensor: feature[1]
        })
        for ele in recall_5_in_10:
            if 1 in ele:
                hit_num += 1
    print("Result-------------------------------------------------------")
    print("recall@%d/10: %f" % (k, float(hit_num)/float(len(features))))
    print("-------------------------------------------------------------")
