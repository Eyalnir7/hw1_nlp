from preprocessing import read_test
from tqdm import tqdm
import numpy as np
from preprocessing import represent_input_with_features


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    sentence = ['*', '*'] + sentence
    tags = feature2id.feature_statistics.tags
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    index_to_tag = {index: tag for index, tag in enumerate(tags)}

    # Create an array with each entry as a zero matrix of size len(tags) x len(tags)
    probs_matrix = np.zeros((n, len(tags), len(tags)), dtype=float)
    args_matrix = np.zeros((n, len(tags), len(tags)), dtype=str)
    start_index = tag_to_index['*']
    probs_matrix[0][start_index][start_index] = 1

    for i in range(1, n):
        last_last_tags, last_tags = beam_search_args(probs_matrix[i - 1], 2)
        for last_tag in len(tags):
            for curr_tag in len(tags):
                if last_tag in last_tags:
                    prob, arg = calc_best_prob(last_last_tags, index_to_tag, tag_to_index, probs_matrix[i - 1],
                                               sentence, pre_trained_weights, feature2id, i, curr_tag, last_tag)
                    probs_matrix[i][curr_tag][last_tag] = prob
                    args_matrix[i][curr_tag][last_tag] = arg

    tags_index = []
    for i in range(n, 0, -1):
        # Get the indices of the maximum value
        max_index = np.argmax(probs_matrix[i])

        # Convert the flattened index to 2D indices
        max_indices = np.unravel_index(max_index, probs_matrix[i].shape)
        if i ==n :
            tags_index.append(max_indices[1])
            tags_index.append(max_indices[0])

        max_arg = args_matrix[max_indices[i]]
        tags_index.append(max_arg)

    tags_values = []
    for i in range(n):
        tags_values.append(index_to_tag[tags_index[n-i-1]])

    return tags_values


def calc_best_prob(last_last_tags, index_to_tag, tag_to_index, probs_matrix, sentence, pre_trained_weights, feature2id,
                   i, curr_tag, last_tag):
    best_prob = 0
    best_arg = None
    for tag in last_last_tags:
        history = (sentence[i], curr_tag, sentence[i - 1], last_tag, sentence[i - 2], tag, sentence[i + 1])
        features_representation = np.array(
            represent_input_with_features(history, feature2id.feature_statistics.feature_to_idx))
        prob = probs_matrix[tag][last_tag] * (features_representation * np.array(pre_trained_weights))
        if prob > best_prob:
            best_prob = prob
            best_arg = tag

    return best_prob, best_arg


def beam_search_args(probs_matrix, b):
    flat_indices = np.argsort(-probs_matrix, axis=None)[:b]
    top_indices = np.unravel_index(flat_indices, probs_matrix.shape)

    # Transpose the indices to get pairs of (i index, j index)
    top_indices_pairs = list(zip(top_indices[0], top_indices[1]))

    return top_indices[0], top_indices[1]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
