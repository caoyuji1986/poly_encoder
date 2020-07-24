import tensorflow as tf

from infer import load_eval_samples


def test_load_eval_samples():

    samples = load_eval_samples(file_name="./dat/test.txt")

    print(samples[:10])


if __name__ == '''__main__''':

    test_load_eval_samples()