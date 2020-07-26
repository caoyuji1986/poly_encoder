import tensorflow as tf

from infer import load_eval_samples, convert_samples_to_features, create_model, main


def test_load_eval_samples():

    samples = load_eval_samples(file_name="./dat/ubuntu/test.txt")
    print(samples[:10])


def test_convert_samples_to_features():

    samples = load_eval_samples(file_name="./dat/ubuntu/test.txt")
    features = convert_samples_to_features(samples=samples, vocab_file='./ckpt/albert_tiny_489k/vocab.txt',
                                           context_length=312, candidate_length=64)
    print(features[:10])


def test_create_model():
    idx, x_context_tensor, x_response_tensor = \
        create_model(model_config_file="./cfg/poly_encoder.json", k=5)
    print(idx)

def test_main():
    main(None)

if __name__ == '''__main__''':

    test_main()