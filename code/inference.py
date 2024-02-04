import scipy
import scipy as sp
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
    n = len(sentence) - 2
    tags = feature2id.feature_statistics.tags
    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    index_to_tag = {index: tag for index, tag in enumerate(tags)}

    # Create an array with each entry as a zero matrix of size len(tags) x len(tags)
    probs_matrix = np.zeros((n, len(tags), len(tags)), dtype=float)
    args_matrix = np.zeros((n, len(tags), len(tags)), dtype=int)
    # probs_matrix = np.full((n, len(tags), len(tags)), -np.inf)
    start_index = tag_to_index['*']
    probs_matrix[0][start_index][start_index] = 1

    for i in range(1, n):
        pp_tags, p_tags = beam_search_args(i, start_index, probs_matrix[i - 1], 3)
        for last_tag in range(len(tags)):
            for curr_tag in range(len(tags)):
                if last_tag in p_tags:
                    prob, arg = calc_best_prob(pp_tags, start_index,probs_matrix[i-1], tags, index_to_tag,
                                               sentence, pre_trained_weights, feature2id, i + 1, curr_tag, last_tag)
                    probs_matrix[i][last_tag][curr_tag] = prob
                    args_matrix[i][last_tag][curr_tag] = arg

    pred_tags = []
    # Get the indices of the maximum value
    max_index = np.argmax(probs_matrix[n - 1])
    # Convert the flattened index to 2D indices
    max_indices = np.unravel_index(max_index, probs_matrix[n - 1].shape)
    curr_tag = max_indices[1]
    p_tag = max_indices[0]
    pred_tags.append(index_to_tag[curr_tag])
    pred_tags.append(index_to_tag[p_tag])

    for i in range(n - 1, 2, -1):
        temp = args_matrix[i][p_tag][curr_tag]
        pred_tags.append(index_to_tag[temp])
        curr_tag = p_tag
        p_tag = temp

    return pred_tags[::-1]


def calc_best_prob(pp_tags, start_index, prev_mat, tags, index_to_tag, sentence, pre_trained_weights, feature2id,
                   i, curr_tag, last_tag):

    if i == 2 or i == 3:
        possible_tags = [start_index]
    else:
        possible_tags = pp_tags

    temp_array = np.zeros(len(tags))
    # best_prob = -np.inf
    best_prob = 0
    best_arg = 0

    for tag in possible_tags:
        history = (sentence[i], index_to_tag[curr_tag], sentence[i - 1], index_to_tag[last_tag], sentence[i - 2],
                   index_to_tag[tag], sentence[i + 1])
        features_representation = represent_input_with_features(history, feature2id.feature_to_idx)
        prob = np.exp(np.sum(np.array([pre_trained_weights[feature] for feature in features_representation])))
        temp_array[tag] = prob
        if prev_mat[tag][last_tag] == 0:
            continue
        prob = prob * prev_mat[tag][last_tag]
        if prob > best_prob:
            best_prob = prob
            best_arg = tag

        # best_prob = best_prob - np.log(np.sum(temp_array))
        best_prob = best_prob / (np.sum(temp_array))

    return best_prob, best_arg

# def calc_best_prob(prev_mat, tags, index_to_tag, sentence, pre_trained_weights, feature2id,
#                    i, curr_tag, last_tag):
#     temp_array = np.ones(len(tags))
#     best_prob = -np.inf
#     best_arg = 0
#
#     for tag in range(len(tags)):
#         history = (sentence[i], index_to_tag[curr_tag], sentence[i - 1], index_to_tag[last_tag], sentence[i - 2],
#                    index_to_tag[tag], sentence[i + 1])
#         features_representation = represent_input_with_features(history, feature2id.feature_to_idx)
#         prob = np.exp(np.sum(np.array([pre_trained_weights[feature] for feature in features_representation])))
#         temp_array[tag] = prob
#         if prev_mat[tag][last_tag] == -np.inf:
#             continue
#         prob = np.log(prob) + prev_mat[tag][last_tag]
#         if prob > best_prob:
#             best_prob = prob
#             best_arg = tag
#
#         best_prob = best_prob - np.log(np.sum(temp_array))
#
#     return best_prob, best_arg


def beam_search_args(i, start_index, probs_matrix, b):
    if i == 1:
        return [start_index], [start_index]

    flat_indices = np.argsort(-probs_matrix, axis=None)[:b]
    top_indices = np.unravel_index(flat_indices, probs_matrix.shape)

    top_indices_pairs = list(zip(top_indices[0], top_indices[1]))

    if i == 2:
        return [start_index], top_indices[1]
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
