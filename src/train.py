# coding:utf-8
import pathlib

import tensorflow as tf

import os

import modeling
import tokenization
from feature import file_based_input_fn_builder, DataProcessor, FeatureThread
from flag_center import FLAGS
from model import PolyEncoderConfig, PolyEncoder, BiEncoder, BiEncoderConfig


def create_input_fn(input_file, is_training, drop_remainder):
    input_fn = file_based_input_fn_builder(input_file=input_file,
                                           is_training=is_training,
                                           drop_remainder=drop_remainder)

    return input_fn


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
    until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def create_train_opt_with_clip(loss, lr_init=0.00005):
    global_steps_ = tf.train.get_or_create_global_step()
    global_step = tf.cast(x=global_steps_, dtype=tf.float32)
    learning_rate = noam_scheme(init_lr=lr_init, global_step=global_step)
    # 论文中使用的就是这个优化器
    admw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
    optimizer = admw(weight_decay=0.0001, learning_rate=learning_rate)
    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, 5.0)
    train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, variables), global_step=global_steps_)
    tf.summary.scalar('learning_rate', learning_rate)
    summaries = tf.summary.merge_all()
    return train_op, learning_rate


def create_train_opt(loss, lr_init=0.001):
    global_steps_ = tf.train.get_or_create_global_step()
    global_step = tf.cast(x=global_steps_, dtype=tf.float32)
    learning_rate = noam_scheme(init_lr=lr_init, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_steps_)
    tf.summary.scalar('learning_rate', learning_rate)
    summaries = tf.summary.merge_all()
    return train_op, learning_rate


def load_weight_from_ckpt(init_checkpoint):
    tvars = tf.trainable_variables()
    assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

def my_model_fn(features, labels, mode, params):

    warmup_steps = min(params['warmup_steps'], params['train_steps'] * 0.1)
    config = params['config']
    ckpt_dir = params['ckpt_dir']
    model_type = params['model_type']
    x, y = features, labels
    if model_type == 'poly-encoder':
        poly_encoder = PolyEncoder(config=config, mode=mode)
    else:
        poly_encoder = BiEncoder(config=config, mode=mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        context_emb, candidate_emb = poly_encoder.create_model(x_context=x, x_response=y)
        loss = poly_encoder.calculate_loss(context_emb_norm=context_emb, candidate_emb_norm=candidate_emb)

        for v in tf.trainable_variables():
            tf.logging.info(v.name)
        load_weight_from_ckpt(ckpt_dir)
        '''
        训练使用了加梯度裁剪的admw
        '''
        #train_op, learning_rate = create_train_opt_with_clip(loss=loss)
        train_op, learning_rate = create_train_opt(loss=loss, lr_init=0.00005)
        hook_dict = {
            #'x_context': x,
            #'x_response': y,
            'loss': loss,
            'learning_rate': learning_rate,
        }
        hook = tf.train.LoggingTensorHook(
            hook_dict,
            every_n_iter=10
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            training_hooks=[hook],
            loss=loss,
            train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:

        raise NotImplementedError('not implemented')

    else:

        raise NotImplementedError('not implemented')


def main(unused_params):

    train_steps = FLAGS.num_train_samples * FLAGS.num_epoches / FLAGS.batch_size
    tf.logging.info('train steps is %d' % train_steps)
    tf.logging.info(str(FLAGS.flag_values_dict()))

    run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                        save_checkpoints_steps=FLAGS.save_checkpoint_steps,
                                        keep_checkpoint_max=FLAGS.keep_checkpoint_max)
    if FLAGS.model_type == 'poly-encoder':
        model_config = PolyEncoderConfig.from_json_file(FLAGS.model_config)
    else:
        model_config = BiEncoderConfig.from_json_file(FLAGS.model_config)
    tf.logging.info(model_config.to_json_string())
    params = {
        'warmup_steps': FLAGS.warmup_steps,
        'train_steps': train_steps,
        'num_epoches': FLAGS.num_epoches,
        'config': model_config,
        'ckpt_dir': FLAGS.ckpt_dir,
        'train_batch_size': FLAGS.batch_size,
        'predict_batch_size': FLAGS.batch_size,
        'model_type': FLAGS.model_type
    }
    estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir,
                                       model_fn=my_model_fn,
                                       config=run_config,
                                       params=params)
    data_processor = DataProcessor()
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)

    if FLAGS.do_train:

        tf_path = os.path.join(FLAGS.data_dir, 'train.success')
        if not os.path.exists(tf_path):
            examples = data_processor.get_train_examples(data_dir=FLAGS.data_dir)

            tasks = FeatureThread.split_task(num_thrd=8,
                                             examples=examples,
                                             out_dir=FLAGS.data_dir,
                                             mode=tf.estimator.ModeKeys.TRAIN)
            thread_inst_list = list()
            for task in tasks:
                thread_inst = FeatureThread(examples=task[0], tokenizer=tokenizer, output_file=task[1])
                thread_inst.start()
                thread_inst_list.append(thread_inst)
            out_file_list = list()
            for i in range(len(thread_inst_list)):
                inst = thread_inst_list[1]
                task = tasks[i]
                out_file_list.append(task[1])
                inst.join()
            pathlib.Path(tf_path).touch()
            tf_path = out_file_list
        else:
            examples = data_processor.get_train_examples(data_dir=FLAGS.data_dir)
            tasks = FeatureThread.split_task(num_thrd=8,
                                             examples=examples,
                                             out_dir=FLAGS.data_dir,
                                             mode=tf.estimator.ModeKeys.TRAIN)
            out_file_list = list()
            for i in range(len(tasks)):
                task = tasks[i]
                out_file_list.append(task[1])
            tf_path = out_file_list
            tf.logging.info("tfrecord list is %s" % ("\n".join(tf_path)))

        tf.logging.info('开始训练 encoder')
        train_input_fn = create_input_fn(input_file=tf_path, is_training=True, drop_remainder=False)
        estimator.train(input_fn=train_input_fn, max_steps=train_steps)

    elif FLAGS.do_eval:
        raise NotImplementedError('其他模式没有实现')

    else:
        raise NotImplementedError('其他模式没有实现')


if __name__ == '''__main__''':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
