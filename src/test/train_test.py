import re
import tensorflow as tf

from model import PolyEncoderConfig, PolyEncoder
from modeling import BertConfig
from train import load_weight_from_ckpt


def test_re():

    m = re.match("^(.*):\\d+$", 'ckpt:1')
    if m is not None:
        name = m.group(1)
        print(name)


def test_load_weight_from_ckpt():

    tf.logging.set_verbosity(tf.logging.INFO)
    x_response_value = [
        [10378, 119, 119, 151, 8815, 8281, 8211, 10425, 8154, 0, 0, 0, 0, 0]
        + [0 for i in range(512 - 14)],
        [165, 8991, 8181, 8184, 131, 120, 120, 8134, 11300, 10540, 8735, 8207, 0, 0]
        + [0 for i in range(512 - 14)]
    ]
    poly_encoder_config = PolyEncoderConfig.from_json_file('../../cfg/poly_encoder.json')
    encoder_inst = PolyEncoder(config=poly_encoder_config, mode=tf.estimator.ModeKeys.TRAIN)

    bert_scope = tf.VariableScope(name="bert", reuse=tf.AUTO_REUSE)
    bert_config = BertConfig.from_json_file(poly_encoder_config.bert_config)
    x_response = tf.convert_to_tensor(value=x_response_value, dtype=tf.int32)
    x_response_emb, x_response_mask = encoder_inst.encode_candidate(x_response=x_response,
                                                                    bert_config=bert_config,
                                                                    bert_scope=bert_scope)
    load_weight_from_ckpt(init_checkpoint="../../ckpt/albert/")


if __name__=="""__main__""":

    test_load_weight_from_ckpt()