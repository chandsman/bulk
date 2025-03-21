import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_scores(hh_df, it_df, seasonal_df=None, parameters=None, version="simple", is_debug=False,
                   add_normal_noise=False):
    if version == "simple":
        return compute_simple_dot_product(hh_df, it_df, seasonal_df=seasonal_df, parameters=parameters,
                                          is_debug=is_debug, add_normal_noise=add_normal_noise)
    elif version == "ml":  # e.g. deep learning model
        return compute_score_from_ml_model(hh_df, it_df, seasonal_df=seasonal_df)
    elif version == "n/a":  # scores provided by user
        return it_df["SCORE"].values
    else:
        raise ValueError("not a valid input for version parameter")


def _compute_score_lr(coefficients, intercept, hh_features, it_features, class_threshold):
    #   print("coefficients shape:", coefficients.shape)
    #   print("hh_features shape:", hh_features.shape)
    #   print("it_features shape:", it_features.shape)
    #   print(coefficients)
    #   print(hh_features)
    #   print(it_features)
    raw_prob_hh = np.dot(coefficients[0:hh_features.shape[1]].reshape(1, hh_features.shape[1]), hh_features.T)
    raw_prob_it = np.dot(
        coefficients[hh_features.shape[1]:hh_features.shape[1] + it_features.shape[1]].reshape(1, it_features.shape[1]),
        it_features.T)

    raw_prob = (raw_prob_hh + raw_prob_it)[0].astype(float) + float(intercept)  # this is correct
    pos_class_prob = 1 / (1 + np.exp(-1.0 * raw_prob))

    is_greater_than_neutral = (pos_class_prob >= 0.5).astype(int)

    # when pos_class_prob = 1.0, then pos_class_to_score = 1.0
    # when pos_class_prob = 0.5, then pos_class_to_score = class_threshold
    # intrapolate score uniformly between 1.0 and class_threshold for pos_class_prob between 1.0 and 0.5
    pos_class_to_score_when_greater = ((1.0 - class_threshold) / 0.5) * (pos_class_prob - 0.5) + class_threshold

    # when pos_class_prob = 0.5, then pos_class_to_score = class_threshold
    # when pos_class_prob = 0.0, then pos_class_to_score = 0.0
    # intrapolate score uniformly between class_threshold and 0.0 for pos_class_prob between 0.5 and 0.0
    pos_class_to_score_when_less = ((class_threshold - 0.0) / 0.5) * pos_class_prob

    pos_class_to_score = is_greater_than_neutral * pos_class_to_score_when_greater + (
                1 - is_greater_than_neutral) * pos_class_to_score_when_less
    return pos_class_to_score.round(4)


def compute_simple_dot_product(hh_df, it_df, seasonal_df=None, parameters=None, is_debug=False, add_normal_noise=False):
    # shelf index in item feature starts from 0
    # in household it starts from SHELF_FEATURE_001 (i.e., 1)
    # so, this setting is correct
    offset = hh_df.columns.tolist().index("shelf_feature_001")  # this is correct
    shelf_column_indices = it_df["shelf_feature_index"].values.astype(int) + offset

    shelf_coeff = 1.0
    if parameters:
        if parameters["shelf_score_coeff"]:
            shelf_coeff = float(parameters["shelf_score_coeff"])
    shelf_scores = (hh_df.iloc[0, :].values[shelf_column_indices]).astype(float).round(4)
    if is_debug:
        print("shelf: ", shelf_coeff)
    scores = shelf_scores * shelf_coeff
    if add_normal_noise:
        rng = np.random.default_rng(seed=int(hh_df["HOUSEHOLD_ID"].values[0]))
        noises = rng.normal(loc=0.0, scale=scores / 20, size=len(scores))
        scores += noises
        del rng, noises

    size_scores = None
    if parameters:
        if parameters["size_score_coeff"]:
            size_coeff = float(parameters["size_score_coeff"])
            offset_size = hh_df.columns.tolist().index("size_feature_0")  # this is correct
            size_column_indices = it_df["size_feature_index"].values.astype(int) + offset_size
            size_scores = (hh_df.iloc[0, :].values[size_column_indices]).astype(float).round(4)
            if is_debug:
                print("size: ", size_coeff)
            scores += size_scores * size_coeff

        price_scores = None
        if parameters["price_score_coeff"]:
            price_coeff = float(parameters["price_score_coeff"])
            offset_price = hh_df.columns.tolist().index("PRICE_FEATURE_0")  # this is correct
            price_column_indices = it_df["PRICE_FEATURE_INDEX"].values.astype(int) + offset_price
            price_scores = (hh_df.iloc[0, :].values[price_column_indices]).astype(float).round(4)
            if is_debug:
                print("price: ", price_coeff)
            scores += price_scores * price_coeff

    seasonal_all_scores = None
    if seasonal_df is not None:
        if parameters:
            if parameters["seasonal_score_coeff"]:
                seasonal_coeff = float(parameters["seasonal_score_coeff"])
                seasonal_all_scores = it_df["BPN_ID"].copy()
                seasonal_all_scores.index = seasonal_all_scores.copy()
                seasonal_all_scores *= 0
                seasonal_all_scores += 10

                seasonal_scores = seasonal_df[["BPN_ID", "SCORE"]].set_index("BPN_ID", inplace=False)["SCORE"]
                intersection = list(set(seasonal_all_scores.index).intersection(seasonal_scores.index))
                seasonal_all_scores.loc[intersection] = seasonal_scores.loc[intersection]
                #seasonal_all_scores = (seasonal_all_scores.values / 1000.0).astype(float).round(4)
                seasonal_all_scores = seasonal_all_scores.astype(float)
                seasonal_all_scores = (seasonal_all_scores.values / 1000.0).round(4)

                if is_debug:
                    print("seasonal: ", seasonal_coeff)
                scores += seasonal_all_scores * seasonal_coeff

    extra_dim = None
    if parameters:
        if parameters["extra_dim"]:
            extra_dim = parameters["extra_dim"]

    added_scores = None
    if extra_dim:
        for feature_group in extra_dim:
            if "embed" == feature_group:
                continue
            if parameters:
                if parameters[f"{feature_group}_score_coeff"]:
                    added_coeff = float(parameters[f"{feature_group}_score_coeff"])
                    prefix_nm = extra_dim[feature_group]["start_nm"]
                    hh_start_offset = hh_df.columns.tolist().index(prefix_nm)
                    prefix_nm = extra_dim[feature_group]["end_nm"]
                    hh_end_offset = hh_df.columns.tolist().index(prefix_nm)

                    prefix_nm = extra_dim[feature_group]["start_nm"]
                    it_start_offset = it_df.columns.tolist().index(prefix_nm)
                    prefix_nm = extra_dim[feature_group]["end_nm"]
                    it_end_offset = it_df.columns.tolist().index(prefix_nm)

                    added_scores = np.dot(
                        hh_df.iloc[0, hh_start_offset:hh_end_offset + 1].values,  # this is correct
                        it_df.iloc[:, it_start_offset:it_end_offset + 1].values.T  # this is correct
                    ).astype(float).round(4)
                    if is_debug:
                        print(f"{feature_group}: ", added_coeff)
                    scores += added_scores * added_coeff

    lr_scores = None
    if parameters:
        if parameters["embed_score_coeff"]:
            coefficients = np.array(parameters["lr_model_coeff"]).astype(float)
            if is_debug:
                print(len(coefficients))
            intercept = float(parameters["lr_model_intercept"])
            class_threshold = float(parameters["lr_model_class_threshold"])
            lr_coeff = float(parameters["embed_score_coeff"])

            if "embed" in extra_dim:
                prefix_nm = extra_dim["embed"]["start_nm"]
                hh_start_offset = hh_df.columns.tolist().index(prefix_nm)
                prefix_nm = extra_dim["embed"]["end_nm"]
                hh_end_offset = hh_df.columns.tolist().index(prefix_nm)

                prefix_nm = extra_dim["embed"]["start_nm"]
                it_start_offset = it_df.columns.tolist().index(prefix_nm)
                prefix_nm = extra_dim["embed"]["end_nm"]
                it_end_offset = it_df.columns.tolist().index(prefix_nm)

                it_features = it_df.iloc[:, it_start_offset:it_end_offset + 1].values
                tmp = hh_df.iloc[0, hh_start_offset:hh_end_offset + 1].values
                hh_features = np.zeros_like(it_features) + tmp
                lr_scores = _compute_score_lr(coefficients, intercept, hh_features, it_features, class_threshold)
                if is_debug:
                    print("embed: ", lr_coeff)
                scores += lr_scores * lr_coeff

    raw_scores = scores.copy().round(4)
    scores = np.round((scores) / (0.00001 + scores.max()), 4)  # to avoid having score = 0
    if is_debug:
        return scores, shelf_scores, size_scores, price_scores, seasonal_all_scores, added_scores, lr_scores, raw_scores
    else:
        return scores


def compute_score_from_ml_model(hh_df, it_df, seasonal_df=None):
    return None


def compute_adjusted_score(dissimilarity, candidate_rec_score):
    # consider 4 cases:
    # W(Dissimilarity), Score, 1-W (Similarity), Dissimilarity, Adjusted Score
    # HIGH              HIGH   LOW               HIGH           HIGHEST (should be always selected since high score and highly dissimilar)
    # HIGH              LOW    LOW               HIGH           MAYBE (very dissimilar but low score)
    # LOW               HIGH   HIGH              LOW            MAYBE (very similar but also high score, so if the score is high enough, it is larger than the second case)
    # LOW               LOW    HIGH              LOW            LOWEST (should not be selected since low score but highly similar)

    weight = dissimilarity
    adjusted_score = weight * candidate_rec_score + (1 - weight) * dissimilarity
    if adjusted_score < 0.5*candidate_rec_score + 0.5*0.5: # (1-weight)*dissimilarity is largest when dissimilarity = 0.5
        adjusted_score = 0.5*candidate_rec_score + 0.5*0.5 + 0.05*(candidate_rec_score+dissimilarity)

    return adjusted_score


def get_avg_dissimilarity(prev_dissimilarity, new_dissimilarity, len_sel_lst, appr_cnt, use_max=True):
    # new dissimilarity func so that max (dissim) between candidate and selected items is used for selection step
    if not use_max:
        w = (1 - new_dissimilarity) * len_sel_lst  # this is correct
        dissimilarity_avg = prev_dissimilarity + new_dissimilarity * w
        # now update for next iteration
        prev_dissimilarity = dissimilarity_avg
        appr_cnt += w

        dissimilarity_avg /= (appr_cnt + 0.00001)
        return dissimilarity_avg, prev_dissimilarity, appr_cnt
    else:
        if appr_cnt == 0:
            dissimilarity_avg = prev_dissimilarity = new_dissimilarity
        else:
            dissimilarity_avg = prev_dissimilarity if new_dissimilarity > prev_dissimilarity else new_dissimilarity
            prev_dissimilarity = dissimilarity_avg
        appr_cnt += 1
    return dissimilarity_avg, prev_dissimilarity, appr_cnt


def get_selected_items_per_hh(
        household_id, target_scores, start_candidate, dissimilarity_dict_pair,
        max_num_select=50, verbose=True, strictly_greedy=True):
    if "target_impression_boost".upper() not in target_scores.columns:
        target_scores["target_impression_boost".upper()] = 0

    candidate_lst, selected_lst, raw_score_lst, impression_boost_lst = target_scores.index.tolist(), [], [], []
    if start_candidate and start_candidate in candidate_lst:
        selected_lst.append(start_candidate)
        candidate_lst.remove(start_candidate)
        raw_score_lst.append(target_scores.loc[start_candidate, "score".upper()])
        impression_boost_lst.append(target_scores.loc[start_candidate, "target_impression_boost".upper()])
    else:
        raise ValueError(f"Can't find start_candidate {start_candidate} in the input list")
    total_selected = 1
    if verbose:
        print(f"Select {selected_lst[-1]} with score {raw_score_lst[-1]} and avg. dissimilarity 0")

    dissimilarity_dict_target_dynamic_prog = {}
    candidates_info = {}
    while total_selected < max_num_select and candidate_lst:
        raw_scores_dict = {}
        impression_boost_dict = {}
        selected_candidate, alternative_candidate, max_adjusted_score, max_candidate_score = None, None, float(
            '-inf'), float('-inf')
        for candidate in candidate_lst:
            if candidate not in candidates_info:
                candidate_raw_score = target_scores.loc[candidate, "score".upper()]
                candidate_impression_boost = target_scores.loc[candidate, "target_impression_boost".upper()]
                candidates_info[candidate] = (candidate_raw_score, candidate_impression_boost)
            else:
                candidate_raw_score, candidate_impression_boost = candidates_info[candidate]

            # get average dissimilarity between this candidate and the items in selected_lst
            if (candidate, selected_lst[-1]) in dissimilarity_dict_pair:
                new_dis_sim = dissimilarity_dict_pair[(candidate, selected_lst[-1])]
            else:
                new_dis_sim = dissimilarity_dict_pair[(selected_lst[-1], candidate)]

            prev_dis_sim, appr_count = 0, 0
            if candidate in dissimilarity_dict_target_dynamic_prog:
                prev_dis_sim, appr_count = dissimilarity_dict_target_dynamic_prog[candidate]

            (dissimilarity_avg,
             prev_dis_sim,
             appr_count
             ) = get_avg_dissimilarity(
                prev_dis_sim, new_dis_sim, len(selected_lst), appr_count
            )
            # update for the next iteration with the same candidate
            dissimilarity_dict_target_dynamic_prog[candidate] = (prev_dis_sim, appr_count)

            adjusted_score = compute_adjusted_score(dissimilarity_avg, candidate_raw_score)

            if selected_candidate:
                if (candidate, selected_candidate) in dissimilarity_dict_pair:
                    dis_sim = dissimilarity_dict_pair[(candidate, selected_candidate)]
                else:
                    dis_sim = dissimilarity_dict_pair[(selected_candidate, candidate)]
            else:
                dis_sim = 1.0

            # keep track of another candidate
            # this alternative candidate is very similar to the current selected candidate
            # it does not have higher dissimilarity than the current selected candidate but it has higher raw score
            if (adjusted_score > max_adjusted_score) or (
                    adjusted_score == max_adjusted_score and candidate_raw_score > max_candidate_score):
                if dis_sim <= 0.1:  # highly similar
                    if alternative_candidate is None or \
                            (alternative_candidate is not None and raw_scores_dict[
                                alternative_candidate] < candidate_raw_score):
                        alternative_candidate = selected_candidate

                max_adjusted_score = adjusted_score
                max_candidate_score = candidate_raw_score
                selected_candidate = candidate
                raw_scores_dict[candidate] = candidate_raw_score
                impression_boost_dict[candidate] = candidate_impression_boost
            else:
                if (adjusted_score < max_adjusted_score) and \
                        (candidate_raw_score > max_candidate_score):
                    if dis_sim <= 0.1:  # highly similar:
                        alternative_candidate = candidate
                        raw_scores_dict[candidate] = candidate_raw_score
                        impression_boost_dict[candidate] = candidate_impression_boost

        # add the selected candidate to selected_lst, and remove it from candidate_lst
        if strictly_greedy or alternative_candidate is None:
            selected_lst.append(selected_candidate)
            raw_score_lst.append(raw_scores_dict[selected_candidate])
            impression_boost_lst.append(impression_boost_dict[selected_candidate])
            candidate_lst.remove(selected_candidate)
        # select the alternative candidate, this one has higher raw score but slightly lower dissimilarity
        else:
            selected_lst.append(alternative_candidate)
            raw_score_lst.append(raw_scores_dict[alternative_candidate])
            impression_boost_lst.append(impression_boost_dict[alternative_candidate])
            candidate_lst.remove(alternative_candidate)
        # now continue until the candidate_lst is empty or
        # the selected_lst has enough items
        total_selected += 1
        if verbose:
            print(
                f"Select {selected_lst[-1]} with score {raw_score_lst[-1]} and max (or avg.) dissimilarity {dissimilarity_avg}")

    data_dict = {
        "household_id": [household_id] * len(selected_lst),
        "selected_bpn": selected_lst,
        "raw_score": raw_score_lst,  # for debugging
        "diversified_order": [i + 1 for i in range(len(selected_lst))],  # the order the list was constructed
        # "target_impression_boost": impression_boost_lst, # for debugging
    }
    columns = ["household_id", "selected_bpn", "raw_score",
               "diversified_order"]  # for debugging "target_impression_boost"
    # columns = ["household_id", "selected_bpn", "diversified_order"] # for prod
    return_df = pd.DataFrame(data_dict, columns=columns)
    # return_df.columns = ["HOUSEHOLD_ID", "BPN_ID", "RANKING"] # for prod
    return_df.columns = ["HOUSEHOLD_ID", "BPN_ID", "RAW_SCORE", "RANKING"]  # for debugging

    return return_df


def rankings_and_diversify(
        household_id, it_df_org, digital_catalog_embedding_df, scores,
        do_diversify=False, dissimilar_threshold=-0.01,
        max_num_diversify=None, verbose=False, is_debug=False,
        sim_catalog_weights={"l2": 0.1, "l3": 0.2, "l4": 0.3, "bpn": 0.4},
        strictly_greedy=False, coeff_for_norm_to_subtract=0.0
):
    it_df = it_df_org.copy()
    it_df["ORG_RANK"] = np.arange(len(it_df)) + 1
    it_df["SCORE"] = scores
    if not do_diversify:
        ranked_df = it_df[["BPN_ID", "ORG_RANK", "SCORE"]].sort_values(["SCORE", "ORG_RANK", "BPN_ID"],
                                                                       ascending=[False, True, False])
        ranked_df["RANKING"] = np.arange(len(ranked_df)) + 1
        ranked_df["HOUSEHOLD_ID"] = household_id
        if is_debug:
            ranked_df["RAW_SCORE"] = ranked_df["SCORE"]  # for debugging
            return ranked_df[["HOUSEHOLD_ID", "BPN_ID", "RAW_SCORE", "RANKING"]]  # for debugging
        else:
            return ranked_df[["HOUSEHOLD_ID", "BPN_ID", "RANKING"]]  # for prod
    else:
        it_df.sort_values(["SCORE", "ORG_RANK", "BPN_ID"], ascending=[False, True, False], inplace=True)
        embed_s, embed_e = None, None
        for i, col in enumerate(it_df):
            if "EMBED_FEATURE" in col:
                if embed_s is None:
                    embed_s = i
                embed_e = i
        embedding_vectors = it_df.iloc[:, embed_s:embed_e + 1].values
        norm_bpn = np.linalg.norm(embedding_vectors, axis=1).reshape((len(embedding_vectors), 1))
        embedding_vectors /= norm_bpn
        dissimilarities = 1 - np.dot(embedding_vectors, embedding_vectors.T)
        it_df["SCORE"] = (scores - coeff_for_norm_to_subtract * norm_bpn).round(4)

        if digital_catalog_embedding_df is not None:
            shelf_lst = set(it_df.loc[:, "SHELF_FEATURE_INDEX"].values)
            it_skim_df = it_df[["BPN_ID", "SHELF_FEATURE_INDEX"]].copy()

            embed_columns = [column for column in digital_catalog_embedding_df.columns if "EMBED" in column]

            conds = np.isin(digital_catalog_embedding_df["SHELF_FEATURE_INDEX"], shelf_lst)
            l2_conds = digital_catalog_embedding_df["LEVEL"].values == "L2"
            l3_conds = digital_catalog_embedding_df["LEVEL"].values == "L3"
            l4_conds = digital_catalog_embedding_df["LEVEL"].values == "L4"

            l2_embed = digital_catalog_embedding_df.loc[
                conds & l2_conds, ["SHELF_FEATURE_INDEX"] + embed_columns].copy()
            l2_embed = l2_embed.merge(it_skim_df, on="SHELF_FEATURE_INDEX", how="right")
            l2_embed.fillna(0.0001, inplace=True)
            l3_embed = digital_catalog_embedding_df.loc[
                conds & l3_conds, ["SHELF_FEATURE_INDEX"] + embed_columns].copy()
            l3_embed = l3_embed.merge(it_skim_df, on="SHELF_FEATURE_INDEX", how="right")
            l3_embed.fillna(0.0001, inplace=True)
            l4_embed = digital_catalog_embedding_df.loc[
                conds & l4_conds, ["SHELF_FEATURE_INDEX"] + embed_columns].copy()
            l4_embed = l4_embed.merge(it_skim_df, on="SHELF_FEATURE_INDEX", how="right")
            l4_embed.fillna(0.0001, inplace=True)

            l2_embedding_vectors = l2_embed.loc[:, embed_columns].values
            l2_embedding_vectors /= np.linalg.norm(l2_embedding_vectors, axis=1).reshape((len(l2_embedding_vectors), 1))
            l2_dissimilarities = 1 - np.dot(l2_embedding_vectors, l2_embedding_vectors.T)
            l3_embedding_vectors = l3_embed.loc[:, embed_columns].values
            l3_embedding_vectors /= np.linalg.norm(l3_embedding_vectors, axis=1).reshape((len(l3_embedding_vectors), 1))
            l3_dissimilarities = 1 - np.dot(l3_embedding_vectors, l3_embedding_vectors.T)
            l4_embedding_vectors = l4_embed.loc[:, embed_columns].values
            l4_embedding_vectors /= np.linalg.norm(l4_embedding_vectors, axis=1).reshape((len(l4_embedding_vectors), 1))
            l4_dissimilarities = 1 - np.dot(l4_embedding_vectors, l4_embedding_vectors.T)

            dissimilarities *= sim_catalog_weights["bpn"]
            dissimilarities += sim_catalog_weights["l2"] * l2_dissimilarities
            dissimilarities += sim_catalog_weights["l3"] * l3_dissimilarities
            dissimilarities += sim_catalog_weights["l4"] * l4_dissimilarities
        dissimilarities = np.round(dissimilarities, 4)

        discard_bpns = set()
        for row_ind in range(dissimilarities.shape[0]):
            discard_by_row = np.where(dissimilarities[row_ind, (row_ind + 1):] <= dissimilar_threshold)[0] + (
                        row_ind + 1)  # this is correct
            if row_ind not in discard_bpns:
                discard_bpns.update(discard_by_row.tolist())
        keep_bpns = [i for i in range(dissimilarities.shape[0]) if i not in discard_bpns]
        discard_bpns = sorted(list(discard_bpns))

        bpn_id_to_indices = {i: bpn for i, bpn in enumerate(it_df["BPN_ID"].values)}
        dissimilarity_pairs = {}
        for i in range(len(bpn_id_to_indices)):
            for j in range(i + 1, len(bpn_id_to_indices)):
                bpn_i, bpn_j = bpn_id_to_indices[i], bpn_id_to_indices[j]
                dissimilarity_pairs[(bpn_i, bpn_j)] = dissimilarities[i, j]

        # since these bpns have low disimilarities with the highly scored bpns, they probably will not be selected early
        not_consider_bpns = it_df.iloc[discard_bpns, :].copy()
        # only do diversification for these bpns
        it_df = it_df.iloc[keep_bpns, :].copy()

        it_df_cp = it_df[["BPN_ID", "SCORE", "ORG_RANK"]].copy()
        it_df_cp.set_index("BPN_ID", inplace=True)
        start_candidate = it_df_cp.iloc[[0], :].index.values[0]
        if not max_num_diversify:
            max_num_diversify = len(it_df_cp)
        ranked_diversify_df = get_selected_items_per_hh(
            household_id, it_df_cp, start_candidate, dissimilarity_pairs,
            max_num_select=max_num_diversify, verbose=verbose, strictly_greedy=strictly_greedy)

        if not is_debug:
            ranked_diversify_df = ranked_diversify_df[["HOUSEHOLD_ID", "BPN_ID", "RANKING"]]

        not_consider_bpns["RANKING"] = np.arange(len(not_consider_bpns)) + 1 + len(it_df_cp)
        not_consider_bpns["HOUSEHOLD_ID"] = household_id
        if is_debug:
            not_consider_bpns["RANKING"] += 100000
            not_consider_bpns["RAW_SCORE"] = not_consider_bpns["SCORE"]  # for debugging
            not_consider_df = not_consider_bpns[["HOUSEHOLD_ID", "BPN_ID", "RAW_SCORE", "RANKING"]]  # for debugging
        else:
            not_consider_df = not_consider_bpns[["HOUSEHOLD_ID", "BPN_ID", "RANKING"]]  # for prod

        if max_num_diversify < len(it_df_cp):
            offset = len(ranked_diversify_df)
            others = it_df_cp.loc[~it_df_cp.index.isin(ranked_diversify_df["BPN_ID"].values), ["SCORE", "ORG_RANK"]]
            others.reset_index(inplace=True)
            others.sort_values(["SCORE", "ORG_RANK", "BPN_ID"], ascending=[False, True, False], inplace=True)
            others["HOUSEHOLD_ID"] = household_id
            others["RANKING"] = np.arange(len(others)) + offset + 1
            if is_debug:
                others["RANKING"] += 10000
                others["RAW_SCORE"] = others["SCORE"]  # for debugging
                others = others[["HOUSEHOLD_ID", "BPN_ID", "RAW_SCORE", "RANKING"]]  # for debugging
            else:
                others = others[["HOUSEHOLD_ID", "BPN_ID", "RANKING"]]  # for prod
            ranked_diversify_df = pd.concat([ranked_diversify_df, others, not_consider_df], axis=0)
        else:
            ranked_diversify_df = pd.concat([ranked_diversify_df, not_consider_df], axis=0)

        return ranked_diversify_df


# def get_digital_embeddings(catalog_embed_tab):
#     # Implementation of get_digital_embeddings function
#     # ... (include the full implementation here)
#     pass

# def get_seasonal(seasonal_tab, ranking_date):
#     # Implementation of get_seasonal function
#     # ... (include the full implementation here)
#     pass
# Load sample data


def e2e_ranking(
        hh_df: pd.DataFrame,
        it_df: pd.DataFrame,
        digital_catalog_embedding_df: Optional[pd.DataFrame] = None,
        seasonal_df: Optional[pd.DataFrame] = None,
        version: str = "simple",
        parameters: Optional[Dict] = None,
        verbose: bool = False,
        do_diversify: bool = False,
        dissimilar_threshold: float = -0.01,
        max_num_diversify: int = 5,
        is_debug: bool = False,
        sim_catalog_weights: Optional[Dict[str, float]] = None,
        strictly_greedy: bool = False,
        coeff_for_norm_to_subtract: float = 0.05,
        add_normal_noise_to_score: bool = True
) -> pd.DataFrame:
    """
    End-to-end ranking function that computes scores and applies diversification.

    Args:
        hh_df (pd.DataFrame): Household features dataframe.
        it_df (pd.DataFrame): Item features dataframe.
        digital_catalog_embedding_df (pd.DataFrame, optional): Digital catalog embeddings.
        seasonal_df (pd.DataFrame, optional): Seasonal scores dataframe.
        version (str, optional): Version of scoring algorithm to use. Defaults to "simple".
        parameters (Dict, optional): Parameters for the scoring algorithm.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        do_diversify (bool, optional): Whether to apply diversification. Defaults to False.
        dissimilar_threshold (float, optional): Threshold for dissimilarity. Defaults to -0.01.
        max_num_diversify (int, optional): Maximum number of items to diversify. Defaults to 5.
        is_debug (bool, optional): Whether to include debug information in output. Defaults to False.
        sim_catalog_weights (Dict[str, float], optional): Weights for similarity calculation.
        strictly_greedy (bool, optional): Whether to use strictly greedy algorithm. Defaults to False.
        coeff_for_norm_to_subtract (float, optional): Coefficient for norm subtraction. Defaults to 0.05.
        add_normal_noise_to_score (bool, optional): Whether to add normal noise to scores. Defaults to True.

    Returns:
        pd.DataFrame: Ranked and (optionally) diversified items.
    """
    if len(hh_df) > 1:
        hh_df = hh_df[hh_df["HOUSEHOLD_ID"].values != 0]

    pinned_df = None
    if "IS_PINNED" in it_df.columns:
        pinned_df = it_df.loc[it_df["IS_PINNED"] == 1, ["BPN_ID"]].copy()
        pinned_df["HOUSEHOLD_ID"] = hh_df["HOUSEHOLD_ID"].values[0]
        pinned_df["RANKING"] = np.arange(len(pinned_df)) + 1
        it_df.loc[it_df["IS_PINNED"] == 1, :].copy()

    if not is_debug:
        scores = compute_scores(hh_df, it_df, seasonal_df=seasonal_df, version=version, parameters=parameters,
                                is_debug=False, add_normal_noise=add_normal_noise_to_score)
    else:
        (scores, shelf_scores, size_scores, price_scores,
         seasonal_all_scores, added_scores, lr_scores, raw_scores) = compute_scores(hh_df, it_df,
                                                                                    seasonal_df=seasonal_df,
                                                                                    version=version,
                                                                                    parameters=parameters,
                                                                                    is_debug=True,
                                                                                    add_normal_noise=add_normal_noise_to_score)
    hhid = hh_df["HOUSEHOLD_ID"].values[0]

    ranked_df = rankings_and_diversify(
        hhid, it_df, digital_catalog_embedding_df, scores,
        do_diversify=do_diversify, dissimilar_threshold=dissimilar_threshold, max_num_diversify=max_num_diversify,
        verbose=verbose, is_debug=is_debug, sim_catalog_weights=sim_catalog_weights,
        strictly_greedy=strictly_greedy, coeff_for_norm_to_subtract=coeff_for_norm_to_subtract
    )

    if is_debug:
        debug_df = it_df[["BPN_ID"]].copy()
        if shelf_scores is not None:
            debug_df["SHELF_SCORE"] = shelf_scores
        if size_scores is not None:
            debug_df["SIZE_SCORE"] = size_scores
        if price_scores is not None:
            debug_df["PRICE_SCORE"] = price_scores
        if seasonal_all_scores is not None:
            debug_df["SEASONAL_SCORE"] = seasonal_all_scores
        if added_scores is not None:
            debug_df["DIET_SCORE"] = added_scores
        if lr_scores is not None:
            debug_df["EMBED_SCORE"] = lr_scores
        debug_df["UNSCALED_SCORE"] = raw_scores
        ranked_df = ranked_df.merge(debug_df, on="BPN_ID", how="left")

    if pinned_df is not None:
        if "RAW_SCORE" in ranked_df.columns:
            pinned_df["RAW_SCORE"] = 10.0
        if is_debug:
            pinned_df["SHELF_SCORE"] = None
            pinned_df["SIZE_SCORE"] = None
            pinned_df["PRICE_SCORE"] = None
            pinned_df["SEASONAL_SCORE"] = None
            pinned_df["DIET_SCORE"] = None
            pinned_df["EMBED_SCORE"] = None
            pinned_df["UNSCALED_SCORE"] = None
        pinned_df = pinned_df[ranked_df.columns.tolist()]
        ranked_df["RANKING"] = ranked_df["RANKING"] + len(pinned_df)
        ranked_df = pd.concat([pinned_df, ranked_df], axis=0)
    return ranked_df