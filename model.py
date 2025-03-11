import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import VotingClassifier
from scipy.special import softmax, expit

class HierarchicalClassifier:
    def __init__(self, multi_classifier, blood_classifier, liver_classifier, psych_classifier, cancer_classifier, 
                 blood_id, liver_id, psych_id, cancer_id, 
                 first_layer_borderline, blood_borderline, liver_borderline, psych_borderline, cancer_borderline, 
                 random_state=3):
        
        self.multi_classifier = multi_classifier
        self.blood_classifier = blood_classifier
        self.liver_classifier = liver_classifier
        self.psych_classifier = psych_classifier
        self.cancer_classifier = cancer_classifier

        self.blood_id = blood_id
        self.liver_id = liver_id
        self.psych_id = psych_id
        self.cancer_id = cancer_id

        self.first_layer_borderline = first_layer_borderline
        self.blood_borderline = blood_borderline
        self.liver_borderline = liver_borderline
        self.psych_borderline = psych_borderline
        self.cancer_borderline = cancer_borderline

        self.seed = random_state
        
    def fit(self, X, y):
        # Labels for the first layer classifier
        y_first_layer = self._create_first_layer_labels(y)
        
        # Train the first layer multi-class classifier
        self._fit_first_layer_classifier(X, y_first_layer, self.multi_classifier, self.first_layer_borderline)
        
        # Train each group's second layer classifier
        self._fit_group_classifier(X, y, self.blood_classifier, self.blood_id, self.blood_borderline)
        self._fit_group_classifier(X, y, self.liver_classifier, self.liver_id, self.liver_borderline)
        self._fit_group_classifier(X, y, self.psych_classifier, self.psych_id, self.psych_borderline)
        self._fit_group_classifier(X, y, self.cancer_classifier, self.cancer_id, self.cancer_borderline)
    
    def predict(self, X):
        # Predict using the first layer multi-class classifier
        first_layer_pred = self.multi_classifier.predict(X)
        
        # Use the corresponding second layer classifier based on the first layer prediction
        final_pred = np.zeros_like(first_layer_pred)
        
        for group_label, classifier, group_ids in zip([0, 1, 2, 3], 
                                                      [self.blood_classifier, self.liver_classifier, self.psych_classifier, self.cancer_classifier],
                                                      [self.blood_id, self.liver_id, self.psych_id, self.cancer_id]):
            mask = (first_layer_pred == group_label)
            if any(mask):
                final_pred[mask] = classifier.predict(X[mask])
                
        return final_pred

    def _create_first_layer_labels(self, y):
        # Create labels for the first layer
        labels = np.full_like(y, -1, dtype=int)  # Default label for safety
        labels[np.isin(y, self.blood_id)] = 0
        labels[np.isin(y, self.liver_id)] = 1
        labels[np.isin(y, self.psych_id)] = 2
        labels[np.isin(y, self.cancer_id)] = 3
        return labels
    
    def _fit_first_layer_classifier(self, X, y, classifier, SMOTE):
        if SMOTE[0]:
            sm = BorderlineSMOTE(k_neighbors=SMOTE[1], m_neighbors=SMOTE[2], n_jobs=-1, random_state=self.seed)
            X_res, y_res = sm.fit_resample(X, y)
        classifier.fit(X_res, y_res)

    def _fit_group_classifier(self, X, y, classifier, group_ids, SMOTE):
        mask = np.isin(y, group_ids)
        if SMOTE[0]:
            sm = BorderlineSMOTE(k_neighbors=SMOTE[1], m_neighbors=SMOTE[2], n_jobs=-1, random_state=self.seed)
            X_res, y_res = sm.fit_resample(X[mask], y[mask])
        classifier.fit(X_res, y_res)
            
    # def predict_proba_or_decision_function(self, clf, X):
    #     """
    #     Predict probabilities using the given classifier and input data. 
    #     First, attempt to call the `predict_proba` method. 
    #     If it fails, compute probabilities using the `decision_function` method and `softmax`.

    #     Parameters:
    #     clf: Classifier instance
    #     X: Input data, shape (n_samples, n_features)

    #     Returns:
    #     Probability predictions
    #     """
    #     try:
    #         # Try to use the predict_proba method
    #         return clf.predict_proba(X)
    #     except AttributeError:
    #         # If predict_proba is not available, use decision_function
    #         try:
    #             decision_scores = clf.decision_function(X)
    #             # Compute probabilities using softmax
    #             return softmax(decision_scores, axis=1)
    #         except AttributeError:
    #             raise AttributeError("The classifier does not have either 'predict_proba' or 'decision_function' methods")

    def predict_proba_or_decision_function(self, clf, X):
        """
        Predict probabilities using the given classifier and input data. 
        First, attempt to call the `predict_proba` method. 
        If it fails, compute probabilities using the `decision_function` method and appropriate method (softmax for multi-class, sigmoid for binary).

        Parameters:
        clf: Classifier instance
        X: Input data, shape (n_samples, n_features)

        Returns:
        Probability predictions
        """
        try:
            # Try to use the predict_proba method
            return clf.predict_proba(X)
        except AttributeError:
            # If predict_proba is not available, use decision_function
            try:
                decision_scores = clf.decision_function(X)
                
                # Check if it's binary or multi-class
                if decision_scores.ndim == 1:
                    # Binary classification: apply sigmoid
                    positive_prob = expit(decision_scores)
                    negative_prob = 1 - positive_prob
                    return np.column_stack((negative_prob, positive_prob))
                elif decision_scores.ndim == 2:
                    # Multi-class classification: apply softmax
                    return softmax(decision_scores, axis=1)
                else:
                    raise ValueError("Unsupported decision function output shape")
            except AttributeError:
                raise AttributeError("The classifier does not have either 'predict_proba' or 'decision_function' methods")

    # def predict_proba(self, X, binary_pair):

    #     # please note that the X cannot be the whole multiclass data set
    #     # because we are going to use predict_proba to create AUROC
    #     # therefore, X should be trimmed into X[binary] as input

    #     first_layer_pred = self.multi_classifier.predict(X)
    #     final_prob = np.zeros((len(first_layer_pred), 2))

    #     for group_label, classifier, group_ids in zip([0, 1, 2, 3], 
    #                                                   [self.blood_classifier, self.liver_classifier, self.psych_classifier, self.cancer_classifier],
    #                                                   [self.blood_id, self.liver_id, self.psych_id, self.cancer_id]):
    #         mask = (first_layer_pred == group_label)

    #         # because we consider probability against healthy, which is in multi-class 0 and subclass 8, which is the first element in blood_id = [8, 10, 12, 17, 19]
    #         if any(mask):
    #             final_prob[mask] = np.column_stack(
    #                 (np.multiply(self.predict_proba_or_decision_function(self.multi_classifier, X[mask])[:, 0], self.predict_proba_or_decision_function(self.blood_classifier, X[mask])[:, 0]), 
    #                  np.multiply(self.predict_proba_or_decision_function(self.multi_classifier, X[mask])[:, group_label], np.max(self.predict_proba_or_decision_function(classifier, X[mask]), axis=1)))
    #                  )

    #     return final_prob
    

    # def predict_proba(self, X, binary_pair):
    #     """
    #     Predicts the probabilities for a binary classification task from a multiclass model.

    #     Parameters:
    #     X (array-like): Input features, preprocessed to only include relevant binary classification data.
    #     binary_pair (tuple): A pair of classes for which the probability needs to be predicted.

    #     Returns:
    #     numpy.ndarray: An array of shape (n_samples, 2) containing the predicted probabilities.
    #     """

    #     # Ensure that X is trimmed to only include the binary classification subset
    #     first_layer_pred = self.multi_classifier.predict(X)
    #     final_prob = np.zeros((len(first_layer_pred), 2))

    #     # Define group labels, classifiers, and IDs in parallel for iteration
    #     group_labels = [0, 1, 2, 3]
    #     classifiers = [self.blood_classifier, self.liver_classifier, self.psych_classifier, self.cancer_classifier]
    #     group_ids = [self.blood_id, self.liver_id, self.psych_id, self.cancer_id]

    #     # Loop through each group label, classifier, and group ID
    #     for group_label, classifier, group_id in zip(group_labels, classifiers, group_ids):
    #         mask = (first_layer_pred == group_label)

    #         # Proceed only if there are samples belonging to the current group label
    #         if np.any(mask):
    #             multi_pred_proba = self.predict_proba_or_decision_function(self.multi_classifier, X[mask])
    #             subclass_pred_proba = self.predict_proba_or_decision_function(classifier, X[mask])

    #             # Probability for the first binary class (healthy against the specific group)
    #             prob_class_0 = np.multiply(multi_pred_proba[:, 0], subclass_pred_proba[:, 0])  # 0 in multi_pred_proba[:, 0] means the 0-th large class, i.e, the blood; 0 in subclass_pred_proba[:, 0] means the 0-th

    #             # Probability for the second binary class (most likely within the specific group)
    #             prob_class_1 = np.multiply(multi_pred_proba[:, group_label], np.max(subclass_pred_proba, axis=1))

    #             # Combine probabilities into the final probability array
    #             final_prob[mask] = np.column_stack((prob_class_0, prob_class_1))

    #     return final_prob


    def predict_proba(self, X):
        """
        Predicts the probabilities for a binary classification task from a multiclass model.

        Parameters:
        X (array-like): Input features, preprocessed to only include relevant binary classification data.
        binary_pair (tuple): A pair of classes for which the probability needs to be predicted.

        Returns:
        numpy.ndarray: An array of shape (n_samples, 2) containing the predicted probabilities.
        """

        # Ensure that X is trimmed to only include the binary classification subset
        first_layer_pred = self.multi_classifier.predict(X)
        final_prob = np.zeros((X.shape[0], 24)) # because we only have 24 phenotypes here

        # Define group labels, classifiers, and IDs in parallel for iteration
        group_labels = [0, 1, 2, 3]
        classifiers = [self.blood_classifier, self.liver_classifier, self.psych_classifier, self.cancer_classifier]
        group_ids = [self.blood_id, self.liver_id, self.psych_id, self.cancer_id]

        # Loop through each group label, classifier, and group ID
        for group_label, classifier, group_id in zip(group_labels, classifiers, group_ids):
            mask = (first_layer_pred == group_label)

            # Proceed only if there are samples belonging to the current group label
            if np.any(mask):
                multi_pred_proba = self.predict_proba_or_decision_function(self.multi_classifier, X[mask])
                multi_pred_proba_1d = multi_pred_proba[:, group_label]
                multi_pred_proba_1d_col = multi_pred_proba_1d[:, np.newaxis]

                subclass_pred_proba = self.predict_proba_or_decision_function(classifier, X[mask])
                
                # mask_rows = np.where(mask)[0]
                # final_prob[mask_rows[:, None], group_id] = multi_pred_proba_1d_col*subclass_pred_proba
                mask_rows = np.where(mask)[0].tolist()
                final_prob[np.ix_(mask_rows, group_id)] = multi_pred_proba_1d_col*subclass_pred_proba

                remaining_columns = [col for col in range(final_prob.shape[1]) if col not in group_id]
                for k, mask_row in enumerate(mask_rows):
                    final_prob[mask_row, remaining_columns] = (1 - multi_pred_proba_1d[k])/(24-len(subclass_pred_proba))

        return final_prob
