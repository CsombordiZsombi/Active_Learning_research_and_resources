import fiftyone.brain as fob


class ScoringFunction:
    def score_sample(self,
                     _sample,
                     _score_tag: str = "score",
                     _prediction_tag: str = "predictions"
                     ):
        pass

    def score_all_samples(self,
                          _pool,
                          _progress_bar=True,
                          _score_tag: str = "score",
                          _prediction_tag: str = "predictions"
                          ):
        if _progress_bar:
            print("Computing scores")
        for sample in _pool.iter_samples(progress=_progress_bar):
            self.score_sample(sample, _score_tag=_score_tag, _prediction_tag=_prediction_tag)


class AverageConfidence(ScoringFunction):
    def score_sample(self,
                     _sample,
                     _score_tag: str = "score",
                     _prediction_tag: str = "predictions"
                     ):

        score_sum = 0
        for detection in _sample[_prediction_tag].detections:
            score_sum += detection.confidence
        num_of_detections = len(_sample[_prediction_tag].detections)
        avg_conf = 0
        if num_of_detections != 0:
            avg_conf = score_sum / num_of_detections
        _sample[_score_tag] = avg_conf
        _sample.save()


class LeastConfident(ScoringFunction):
    def score_sample(self,
                     _sample,
                     _score_tag: str = "score",
                     _prediction_tag: str = "predictions"
                     ):
        min_conf = 1
        for detection in _sample[_prediction_tag].detections:
            min_conf = min(min_conf, detection.confidence)
        _sample[_score_tag] = min_conf
        _sample.save()


class MostConfident(ScoringFunction):
    def score_sample(self,
                     _sample,
                     _score_tag: str = "score",
                     _prediction_tag: str = "predictions"
                     ):
        max_conf = 0
        for detection in _sample[_prediction_tag].detections:
            max_conf = max(max_conf, detection.confidence)
        _sample[_score_tag] = max_conf
        _sample.save()


class UniquenessScore(ScoringFunction):
    def score_all_samples(self,
                          _pool,
                          _progress_bar=True,
                          _score_tag: str = "score",
                          _prediction_tag: str = "predictions"
                          ):
        fob.compute_uniqueness(_pool, uniqueness_field=_score_tag)