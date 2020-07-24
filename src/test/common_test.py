import tensorflow as tf

from operation import tensor_norm


def test_tensor_norm():

    a = tf.convert_to_tensor(value=[[1,2,3,4],[5,6,7,8]], dtype=tf.float32)
    b = tensor_norm(a)
    sess = tf.Session()
    print(sess.run(b))


if __name__ == '__main__':
    test_tensor_norm()
