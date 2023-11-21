from scoringfunction import ScoringFunction
from operator import attrgetter


class SamplingStrategy:
    def get_n_samples(self,
                      _pool,
                      _scoring_function: ScoringFunction,
                      _n_samples: int,
                      _score_tag: str = "score",
                      _prediction_tag: str = "predictions",
                      _progression_bar=True,
                      _delete_from_pool=True
                      ):
        pass


class RandomSamples(SamplingStrategy):
    def get_n_samples(self,
                      _pool,
                      _scoring_function: ScoringFunction,
                      _n_samples: int,
                      _score_tag: str = "score",
                      _prediction_tag: str = "predictions",
                      _progression_bar=True,
                      _delete_from_pool=True
                      ):
        data_points_to_query = _pool.take(_n_samples)
        if _delete_from_pool:
            _pool.delete_samples(data_points_to_query)
            _pool.save()
        return _pool, data_points_to_query


class MinScore(SamplingStrategy):
    def get_n_samples(self,
                      _pool,
                      _scoring_function: ScoringFunction,
                      _n_samples: int,
                      _score_tag: str = "score",
                      _prediction_tag: str = "predictions",
                      _progression_bar=True,
                      _delete_from_pool=True
                      ):

        data_points_to_query = []
        _scoring_function.score_all_samples(_pool, _score_tag=_score_tag, _prediction_tag=_prediction_tag,
                                            _progress_bar=_progression_bar)
        if _progression_bar:
            print("Sampling")
        for _sample in _pool.iter_samples(progress=_progression_bar):
            if len(data_points_to_query) < _n_samples:
                data_points_to_query.append(_sample)
            else:
                max_sample = max(data_points_to_query, key=attrgetter(_score_tag))
                if max_sample[_score_tag] > _sample[_score_tag]:
                    data_points_to_query.remove(max_sample)
                    data_points_to_query.append(_sample)
        if _delete_from_pool:
            _pool.delete_samples(data_points_to_query)
            _pool.save()
        return _pool, data_points_to_query


class MaxScore(SamplingStrategy):
    def get_n_samples(self,
                      _pool,
                      _scoring_function: ScoringFunction,
                      _n_samples: int,
                      _score_tag: str = "score",
                      _prediction_tag: str = "predictions",
                      _progression_bar=False,
                      _delete_from_pool=True
                      ):

        data_points_to_query = []
        _scoring_function.score_all_samples(_pool, _score_tag=_score_tag, _prediction_tag=_prediction_tag,
                                            _progress_bar=_progression_bar)
        if _progression_bar:
            print("Sampling")
        for _sample in _pool.iter_samples(progress=_progression_bar):
            if len(data_points_to_query) < _n_samples:
                data_points_to_query.append(_sample)
            else:
                min_sample = min(data_points_to_query, key=attrgetter(_score_tag))
                if min_sample[_score_tag] < _sample[_score_tag]:
                    data_points_to_query.remove(min_sample)
                    data_points_to_query.append(_sample)
        if _delete_from_pool:
            _pool.delete_samples(data_points_to_query)
            _pool.save()
        return _pool, data_points_to_query