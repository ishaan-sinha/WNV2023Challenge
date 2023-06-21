import timeit
import numpy as np


def interval_score(
        observations,
        alpha,
        q_dict=None,
        q_left=None,
        q_right=None,
        percent=False,
        check_consistency=True,
):

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(round(alpha / 2, 2))
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha / 2}-quantile")

        q_right = q_dict.get(round(1 - (alpha / 2), 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1 - (alpha / 2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
            (
                    np.clip(q_left - observations, a_min=0, a_max=None)
                    + np.clip(observations - q_right, a_min=0, a_max=None)
            )
            * 2
            / alpha
    )
    if percent:
        sharpness = sharpness / np.abs(observations)
        calibration = calibration / np.abs(observations)
    total = sharpness + calibration
    return total, sharpness, calibration

def weighted_interval_score(
        observations, alphas, q_dict, weights=None, percent=False, check_consistency=False
):

    if weights is None:
        weights = np.array(alphas) / 2


    def weigh_scores(tuple_in, weight):
        return tuple_in[0] * weight, tuple_in[1] * weight, tuple_in[2] * weight

    interval_scores = [
        i
        for i in zip(
            *[
                weigh_scores(
                    interval_score(
                        observations,
                        alpha,
                        q_dict=q_dict,
                        percent=percent,
                        check_consistency=check_consistency,
                    ),
                    weight,
                )
                for alpha, weight in zip(alphas, weights)
            ]
        )
    ]

    total = np.sum(np.vstack(interval_scores[0]), axis=0) / sum(weights)
    sharpness = np.sum(np.vstack(interval_scores[1]), axis=0) / sum(weights)
    calibration = np.sum(np.vstack(interval_scores[2]), axis=0) / sum(weights)

    return total, sharpness, calibration


observations_test = np.array([1, 2, 3, 4, 5, 6, 7, 8])
quantile_dict_test = {
    0.1: np.array([2, 3  , 5   , 9  , 1  , -3  , 0.2, 8.7]),
    0.2: np.array([2, 4.6, 5   , 9.4, 1.4, -2  , 0.4, 8.8]),
    0.5: np.array([2, 4.7, 5.2 , 9.6, 1.8, -2  , 0.4, 8.8]),
    0.8: np.array([4, 4.8, 5.7 , 12 , 4.3, -1.5, 2  , 8.9]),
    0.9: np.array([5, 5  , 7   , 13 , 5  , -1  , 3  , 9])
}
print(sum(sum(weighted_interval_score(observations_test, alphas=[0.2, 0.4, 1.0, 1.6, 1.8], q_dict=quantile_dict_test))))




