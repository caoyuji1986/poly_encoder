import copy
import json

import six
import tensorflow as tf

import modeling
from modeling import BertConfig
from operation import get_shape_list, scaled_dot_product_attention, tensor_norm


class BaseEncoder:

    def make_mask_by_value(self, x):
        '''
        :param x: tensor with dtype as tf.int32
        :return: [1,1,...,1,0,0,...,0]
        '''
        zeros = tf.zeros_like(tensor=x, dtype=tf.int32)
        ones = tf.ones_like(tensor=x, dtype=tf.int32)
        x_mask = tf.where(condition=tf.equal(x=x, y=zeros), x=zeros, y=ones)
        return x_mask

    def calculate_loss(self, context_emb_norm, candidate_emb_norm):
        # batch_size x batch_size
        logit_matrix = tf.matmul(a=context_emb_norm, b=candidate_emb_norm, transpose_b=True)
        log_softmax_matrix = - tf.nn.log_softmax(logits=logit_matrix, axis=-1)
        batch_size = get_shape_list(context_emb_norm)[0]
        mask_matrix = tf.eye(num_rows=batch_size)
        loss_matrix = log_softmax_matrix * mask_matrix

        loss_per_sample = tf.reduce_sum(input_tensor=loss_matrix, axis=-1)
        loss = tf.reduce_mean(input_tensor=loss_per_sample)

        return loss

    def calculate_distance(self, context_emb_norm, candidate_emb_norm):

        tmp = tf.multiply(x=context_emb_norm, y=candidate_emb_norm)
        cosine_distance = tf.reduce_sum(input_tensor=tmp, axis=-1)
        return cosine_distance


class BiEncoderConfig(object):
    """Configuration for `TransformerModel`."""

    def __init__(self, bert_config='./ckpt/albert/albert_config_base.json',
                 max_seq_length=512):

        self.bert_config = bert_config
        self.max_seq_length = max_seq_length

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = BiEncoderConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BiEncoder(BaseEncoder):

    def __init__(self, config, mode):

        self._config = config
        self._mode = mode

    def encode_context(self, x_context, bert_config, bert_scope):

        return self.__encode_context(x_context=x_context, bert_config=bert_config, scope=bert_scope)

    def encode_candidate(self, x_response, bert_config, bert_scope):

        return self.__encode_candidate(x_response=x_response, bert_config=bert_config, scope=bert_scope)

    def __encode_context(self, x_context, bert_config, scope):

        x_mask = tf.cast(x=self.make_mask_by_value(x=x_context), dtype=tf.float32)
        x_segment = tf.zeros_like(tensor=x_context, dtype=tf.int32, name='segments')
        """
        Creates a classification model.
        """
        model = modeling.BertModel(
            config=bert_config,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            input_ids=x_context,
            input_mask=x_mask,
            token_type_ids=x_segment,
            use_one_hot_embeddings=False,
            scope=scope)
        # batch_size x hidden_size
        x_context_hidden = model.get_pooled_output()

        return x_context_hidden

    def __encode_candidate(self, x_response, bert_config, scope):

        x_response_mask = tf.cast(x=self.make_mask_by_value(x=x_response), dtype=tf.float32)
        x_response_segment = tf.zeros_like(tensor=x_response, dtype=tf.int32, name='x_response_segments')
        """
        Creates a classification model.
        """
        model_response = modeling.BertModel(
            config=bert_config,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            input_ids=x_response,
            input_mask=x_response_mask,
            token_type_ids=x_response_segment,
            use_one_hot_embeddings=False,
            scope=scope)
        # batch_size x hidden_size
        x_response_emb = model_response.get_pooled_output()

        return x_response_emb

    def create_model(self, x_context, x_response):

        bert_scope = tf.VariableScope(name="bert", reuse=tf.AUTO_REUSE)
        bert_config = BertConfig.from_json_file(self._config.bert_config)

        context_emb = self.__encode_context(x_context=x_context, bert_config=bert_config, scope=bert_scope)
        candidate_emb = self.__encode_candidate(x_response=x_response, bert_config=bert_config, scope=bert_scope)

        context_emb_norm = self.__normalize_emb(context_emb=context_emb,
                                                name='context',
                                                max_len=self._config.max_seq_length)
        x_response_hidden_norm = self.__normalize_emb(context_emb=candidate_emb,
                                                      name='candidate',
                                                      max_len=self._config.max_seq_length)

        return context_emb_norm, x_response_hidden_norm

    def __normalize_emb(self, context_emb, name, max_len):

        context_emb = context_emb
        context_emb_norm = modeling.layer_norm(input_tensor=context_emb)
        context_emb_norm = tf.layers.dense(inputs=context_emb_norm, units=64, name=name + 'prj')
        context_emb_norm = tensor_norm(tensor=context_emb_norm)
        return context_emb_norm


class PolyEncoderConfig(object):
    """Configuration for `TransformerModel`."""

    def __init__(self,
                 code_num=6,
                 attention_dropout_prob=0.1,
                 bert_config='./ckpt/albert/albert_config_base.json'):

        self.code_num = code_num
        self.attention_dropout_prob = attention_dropout_prob
        self.bert_config = bert_config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        config = PolyEncoderConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class PolyEncoder(BaseEncoder):

    def __init__(self, config, mode):
        self._config = config
        self._mode = mode

    def make_mask_by_value(self, x):
        '''
        :param x: tensor with dtype as tf.int32
        :return: [1,1,...,1,0,0,...,0]
        '''
        zeros = tf.zeros_like(tensor=x, dtype=tf.int32)
        ones = tf.ones_like(tensor=x, dtype=tf.int32)
        x_mask = tf.where(condition=tf.equal(x=x, y=zeros), x=zeros, y=ones)
        return x_mask

    def encode_context(self, x_context, bert_config, bert_scope):

        return self.__encode_context(x_context=x_context, bert_config=bert_config, bert_scope=bert_scope)

    def encode_candidate(self, x_response, bert_config, bert_scope):

        return self.__encode_candidate(x_response=x_response, bert_config=bert_config, bert_scope=bert_scope)

    def __encode_context(self, x_context, bert_config, bert_scope):

        batch_size = get_shape_list(x_context)[0]
        x_mask = tf.cast(x=self.make_mask_by_value(x=x_context), dtype=tf.float32)
        x_segment = tf.zeros_like(tensor=x_context, dtype=tf.int32, name='segments')
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            input_ids=x_context,
            input_mask=x_mask,
            token_type_ids=x_segment,
            use_one_hot_embeddings=False,
            scope=bert_scope)
        # batch_size x seq_length1 x hidden_size
        x_context_hidden = model.get_sequence_output()

        poly_code = tf.range(start=1, limit=self._config.code_num+1)
        poly_code_tmp = tf.tile(input=poly_code, multiples=[batch_size])
        poly_codes = tf.reshape(tensor=poly_code_tmp, shape=[batch_size, self._config.code_num])
        poly_code_mask = tf.cast(x=self.make_mask_by_value(x=poly_codes), dtype=tf.float32)
        code_embedding = tf.get_variable(name='code_embedding',
                                         shape=[self._config.code_num + 1, bert_config.hidden_size],
                                         dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        # batch_size x code_num x hidden_size
        poly_code_embs = tf.nn.embedding_lookup(params=code_embedding, ids=poly_codes)
        # batch_size x code_num x hidden_size
        x_context_vecs = scaled_dot_product_attention(
            q=poly_code_embs, k=x_context_hidden, v=x_context_hidden,
            mask_q=poly_code_mask, mask_k=x_mask, mask_v=x_mask,
            attention_dropout=self._config.attention_dropout_prob,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            attention_future=True, dk=1)

        return x_context_vecs,  poly_code_mask

    def __encode_candidate(self, x_response, bert_config, bert_scope):

        batch_size = get_shape_list(x_response)[0]
        x_response_mask = tf.cast(x=self.make_mask_by_value(x=x_response), dtype=tf.float32)
        x_response_segment = tf.zeros_like(tensor=x_response, dtype=tf.int32, name='x_response_segments')
        """Creates a classification model."""
        model_response = modeling.BertModel(
            config=bert_config,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            input_ids=x_response,
            input_mask=x_response_mask,
            token_type_ids=x_response_segment,
            use_one_hot_embeddings=False,
            scope=bert_scope)
        # batch_size x seq_length x hidden_size
        x_response_emb = model_response.get_sequence_output()

        poly_codes = tf.zeros(shape=[batch_size, 1], dtype=tf.int32)
        poly_code_mask = tf.ones_like(tensor=poly_codes, dtype=tf.float32)
        code_embedding = tf.get_variable(name='candidate_code_embedding',
                                         shape=[1, bert_config.hidden_size],
                                         dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        # batch_size x 1 x hidden_size
        poly_code_embs = tf.nn.embedding_lookup(params=code_embedding, ids=poly_codes)

        # batch_size x 1 x hidden_size
        x_response_vecs = scaled_dot_product_attention(
            q=poly_code_embs, k=x_response_emb, v=x_response_emb,
            mask_q=poly_code_mask, mask_k=x_response_mask, mask_v=x_response_mask,
            attention_dropout=self._config.attention_dropout_prob,
            is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
            attention_future=True, dk=1)

        return x_response_vecs, poly_code_mask

    def __normalize_emb(self, context_emb, name):

        context_emb = context_emb
        #context_emb_norm = modeling.layer_norm(input_tensor=context_emb)
        context_emb_norm = tf.layers.dense(inputs=context_emb, units=64, name=name + 'prj')
        context_emb_norm = tensor_norm(tensor=context_emb_norm)
        return context_emb_norm
    #
    # def __normalize_emb(self, context_emb, name, max_len):
    #
    #     (context_emb, x_mask) = context_emb
    #     batch_size, seq_len, hidden_size = get_shape_list(context_emb)
    #     context_emb_norm = modeling.layer_norm(input_tensor=context_emb)
    #     x_mask_norm = tf.expand_dims(input=x_mask, axis=-1)
    #     # batch_size x seq_len x hidden_size
    #     context_emb_norm = context_emb * x_mask_norm
    #     context_emb_norm = tf.reshape(tensor=context_emb_norm,
    #                                   shape=[batch_size, seq_len * hidden_size])
    #
    #     proj_weight = tf.get_variable(name=name + "prj_w", dtype=tf.float32, shape=[max_len, hidden_size, 64])
    #     proj_bias = tf.get_variable(name=name + "prj_b", dtype=tf.float32, shape=[64])
    #     proj_weight = tf.reshape(tensor=proj_weight[:seq_len, :, :], shape=[seq_len*hidden_size, 64])
    #     context_emb_norm = tf.nn.bias_add(value=tf.matmul(a=context_emb_norm, b=proj_weight), bias=proj_bias)
    #
    #     context_emb_norm = tensor_norm(tensor=context_emb_norm)
    #     return context_emb_norm

    def create_model(self, x_context, x_response):

        bert_scope = tf.VariableScope(name="bert", reuse=tf.AUTO_REUSE)
        bert_config = BertConfig.from_json_file(self._config.bert_config)
        # 构造 context/response的编码输出
        # >> 上下文编码
        # batch_size x code_num x hidden_size
        context_vecs, context_vecs_mask = self.__encode_context(x_context=x_context,
                                                                bert_config=bert_config,
                                                                bert_scope=bert_scope)
        # >> 构造response编码
        # batch_size x 1 x hidden_size
        x_response_emb, x_response_mask = self.__encode_candidate(x_response=x_response,
                                                                  bert_config=bert_config,
                                                                  bert_scope=bert_scope)
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            batch_size = get_shape_list(x_context)[0]
            x_response_emb = tf.reshape(tensor=x_response_emb, shape=[batch_size, bert_config.hidden_size])
            # batch_size x batch_size x hidden_size
            x_response_emb = tf.reshape(tensor=tf.tile(input=x_response_emb, multiples=[batch_size,  1]),
                                        shape=[batch_size, batch_size, bert_config.hidden_size])

        # >> 构建response/context的交互编码
        # batch_size x batch_size x hidden_size if train else batch_size x 1 x hidden_size
        context_emb = scaled_dot_product_attention(q=x_response_emb, k=context_vecs, v=context_vecs,
                                                   mask_q=x_response_mask, mask_k=context_vecs_mask, mask_v=context_vecs_mask,
                                                   attention_dropout=self._config.attention_dropout_prob,
                                                   is_training=self._mode == tf.estimator.ModeKeys.TRAIN,
                                                   attention_future=True, dk=1
                                                   )
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            context_emb_norm = self.__normalize_emb(context_emb=context_emb, name='context')
            candidate_emb_norm = self.__normalize_emb(context_emb=x_response_emb, name='response')
            return context_emb_norm, candidate_emb_norm

        batch_size = get_shape_list(x_response)[0]
        context_emb = tf.reshape(tensor=context_emb, shape=[batch_size, bert_config.hidden_size])
        x_response_emb = tf.reshape(tensor=x_response_emb, shape=[batch_size, bert_config.hidden_size])
        # 为构建score准备归一化向量
        context_emb_norm = self.__normalize_emb(context_emb=context_emb, name='context')
        candidate_emb_norm = self.__normalize_emb(context_emb=x_response_emb, name='response')

        return context_emb_norm, candidate_emb_norm

    def calculate_loss(self, context_emb_norm, candidate_emb_norm):
        # batch_size x batch_size
        logit_matrix = tf.reduce_sum(input_tensor=(tf.multiply(x=context_emb_norm, y=candidate_emb_norm)), axis=-1)
        log_softmax_matrix = - tf.nn.log_softmax(logits=logit_matrix * 5, axis=-1)
        batch_size = get_shape_list(context_emb_norm)[0]
        mask_matrix = tf.eye(num_rows=batch_size)
        loss_matrix = log_softmax_matrix * mask_matrix

        loss_per_sample = tf.reduce_sum(input_tensor=loss_matrix, axis=-1)
        loss = tf.reduce_mean(input_tensor=loss_per_sample)

        return loss


