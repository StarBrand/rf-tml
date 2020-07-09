"""random_forest.py: Random Forest training and prediction"""

from models import Model
from models.random_forest import CONFIG
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    """
    Random Forest Class

    :ivar classifier: A RandomForestClassifier
    :ivar name: Name to identify test
    """

    def __init__(self, name: str) -> None:
        """
        Generates instance of Random Forest

        :param name: Name to be used as id
        :return:
        """
        super().__init__(name)
        self.classifier = RandomForestClassifier(n_estimators=CONFIG["n"],
                                                 n_jobs=CONFIG["jobs"],
                                                 random_state=CONFIG["seed"])
