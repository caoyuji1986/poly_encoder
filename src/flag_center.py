# coding:utf8

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name='model_dir', default='./out/model/', help='模型的位置')
flags.DEFINE_string(name='data_dir', default='./dat/wnt', help='训练和测试数据的位置')
flags.DEFINE_string(name='ckpt_dir', default='./ckpt/albert_tiny_489k', help='bert 模型的位置')

flags.DEFINE_boolean(name='do_predict', default=False, help='是否开始预测')
flags.DEFINE_boolean(name='do_train', default=True, help='是否开始训练')

flags.DEFINE_integer(name='save_checkpoint_steps', default=1000, help='保存ckpt的训练步数')
flags.DEFINE_integer(name='keep_checkpoint_max', default=5, help='最多保存多少ckpt')
flags.DEFINE_string(name='model_config', default='./cfg/transformer.json', help='模型配置位置')

flags.DEFINE_integer(name='num_train_samples', default=6000, help='训练样本数量')
flags.DEFINE_integer(name='warmup_steps', default=4000, help='预热步数')
flags.DEFINE_integer(name='num_epoches', default=40, help='epoches 的数量')
flags.DEFINE_integer(name='batch_size', default=256, help='batch size')
flags.DEFINE_string(name='vocab_file', default="", help='词表文件')

flags.DEFINE_integer(name='context_length', default=128, help='上下文的最大长度')
flags.DEFINE_integer(name='candidate_length', default=64, help='回复的最大长度')

flags.DEFINE_string(name="model_type", default="bi-encoder", help='模型名称')
