from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
import sys
from src.exception import CustomException



def initiate_model_trainer(self, train_array, test_array, target_columns):
    try:
        # Split features & targets
        X_train = train_array[:, :-len(target_columns)]
        y_train = train_array[:, -len(target_columns):]

        X_test = test_array[:, :-len(target_columns)]
        y_test = test_array[:, -len(target_columns):]

        # 1️⃣ Baseline models
        models = {
            "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=2000)),
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()),
            "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=500, max_depth=20)),
            "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier()),
            "SVM": MultiOutputClassifier(SVC(probability=True)),
            "XGBoost": MultiOutputClassifier(XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)),
            "CatBoost": MultiOutputClassifier(CatBoostClassifier(verbose=0)),
            "Classifier Chain RF": ClassifierChain(RandomForestClassifier(n_estimators=300, max_depth=15)),
        }

        # 2️⃣ Train baseline models
        model_report, trained_models = self.evaluate_models(X_train, y_train, X_test, y_test, models)

        # 3️⃣ Hyperparameter tuning for XGB & CatBoost
        tuned_models = self.hyperparameter_tuning(X_train, y_train)
        tuned_report, tuned_trained = self.evaluate_models(X_train, y_train, X_test, y_test, tuned_models)

        # Merge results
        model_report.update(tuned_report)
        trained_models.update(tuned_trained)

        # 4️⃣ Build Ensembles (Voting + Stacking)
        voting_clf = MultiOutputClassifier(
            VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=500, max_depth=20)),
                    ("xgb", XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)),
                    ("cat", CatBoostClassifier(verbose=0)),
                ],
                voting="soft"
            )
        )
        stacking_clf = MultiOutputClassifier(
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=500, max_depth=20)),
                    ("xgb", XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)),
                ],
                final_estimator=LogisticRegression(max_iter=2000),
                cv=3
            )
        )

        ensemble_models = {
            "Voting Ensemble": voting_clf,
            "Stacking Ensemble": stacking_clf,
        }
        ensemble_report, ensemble_trained = self.evaluate_models(X_train, y_train, X_test, y_test, ensemble_models)

        model_report.update(ensemble_report)
        trained_models.update(ensemble_trained)

        # 5️⃣ Pick best by F1
        best_model_name = max(model_report, key=lambda name: model_report[name]["f1_score"])
        best_model = trained_models[best_model_name]
        best_score = model_report[best_model_name]["f1_score"]

        # 6️⃣ Threshold tuning
        best_model, thresholds = self.threshold_tuning(best_model, X_test, y_test)

        # 7️⃣ Save final best model
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj={"model": best_model, "thresholds": thresholds},
        )

        return model_report, best_model_name, best_score

    except Exception as e:
        raise CustomException(e, sys)
