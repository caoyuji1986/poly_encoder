import tensorflow as tf

from modeling import BertConfig

tf.enable_eager_execution()

from model import PolyEncoderConfig, PolyEncoder


def test_PolyEncoderConfig():

    poly_encoder_config = PolyEncoderConfig.from_json_file('../../cfg/poly_encoder.json')
    print(poly_encoder_config.to_json_string())


class TestPolyEncoder:

    def test_setup(self):

        self._poly_encoder_config = PolyEncoderConfig.from_json_file('../../cfg/poly_encoder.json')
        self._encoder_inst = PolyEncoder(config=self._poly_encoder_config, mode=tf.estimator.ModeKeys.TRAIN)

    def test_encode_context(self):

        x_context_value = [
            [ 151, 12553, 8997, 8792,  10086, 8168, 10481, 9356,  8174, 10404, 9066, 10003, 10610, 10879]
            + [0 for i in range(312 - 14)],
            [8670, 11136, 8997, 10564, 8303,  8228, 8373,  10003, 8307, 119,   151,  12553, 8233,  8815]
            + [0 for i in range(312 - 14)]
        ]

        bert_scope = tf.VariableScope(name="bert", reuse=tf.AUTO_REUSE)
        bert_config = BertConfig.from_json_file(self._poly_encoder_config.bert_config)
        x_context = tf.convert_to_tensor(value=x_context_value, dtype=tf.int32)
        context_vecs, poly_code_mask = self._encoder_inst.encode_context(x_context=x_context,
                                                                                   bert_config=bert_config,
                                                                                   bert_scope=bert_scope)
        print(context_vecs)
        print(poly_code_mask)

    def test_encode_candidate(self):

        x_response_value = [
            [10378, 119, 119, 151, 8815, 8281, 8211, 10425, 8154, 0, 0, 0, 0, 0]
            + [0 for i in range(512 - 14)],
            [165, 8991, 8181, 8184, 131, 120, 120, 8134, 11300, 10540, 8735, 8207, 0, 0]
            + [0 for i in range(512 - 14)]
        ]

        bert_scope = tf.VariableScope(name="bert", reuse=tf.AUTO_REUSE)
        bert_config = BertConfig.from_json_file(self._poly_encoder_config.bert_config)
        x_response = tf.convert_to_tensor(value=x_response_value, dtype=tf.int32)
        x_response_emb, x_response_mask = self._encoder_inst.encode_candidate(x_response=x_response,
                                                                              bert_config=bert_config,
                                                                              bert_scope=bert_scope)
        print(x_response_emb)
        print(x_response_mask)

    def test_create_model(self):

        x_context_value = [
            [ 151, 12553, 8997, 8792,  10086, 8168, 10481, 9356,  8174, 10404, 9066, 10003, 10610, 10879]
            + [0 for i in range(312 - 14)],
            [8670, 11136, 8997, 10564, 8303,  8228, 8373,  10003, 8307, 119,   151,  12553, 8233,  8815]
            + [0 for i in range(312 - 14)]
        ]
        x_response_value = [
            [10378,119,   119,  151,   8815,  8281, 8211,  10425, 8154, 0,     0,     0,    0,     0]
            + [0 for i in range(112 - 14)],
            [165,  8991,  8181, 8184,  131,   120,  120,   8134,  11300,10540, 8735,  8207, 0,     0]
            + [0 for i in range(112 - 14)]
        ]
        x_context = tf.convert_to_tensor(value=x_context_value, dtype=tf.int32)
        x_response = tf.convert_to_tensor(value=x_response_value, dtype=tf.int32)
        context_emb, candidate_emb = self._encoder_inst.create_model(x_context=x_context, x_response=x_response)
        print(context_emb)
        print(candidate_emb)

        return context_emb, candidate_emb

    def test_calculate_loss(self):

        context_emb, candidate_emb = test_poly_encoder.test_create_model()
        loss = self._encoder_inst.calculate_loss_distance(context_emb=context_emb, candidate_emb=candidate_emb)
        print(loss)

if __name__ == """__main__""":

    test_poly_encoder = TestPolyEncoder()
    test_poly_encoder.test_setup()

    test_poly_encoder.test_calculate_loss()
