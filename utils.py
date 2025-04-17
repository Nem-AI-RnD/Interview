from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import root_scalar


class SpamDataset:
    """
    A class to generate a synthetic dataset for a spam classification task.

    The dataset contains emails with features that are commonly used
    in spam classification. The dataset includes 9 covariates and a binary response.

    Feature Descriptions:
      - num_links: The number of hyperlinks in the email. Spam emails
                   often contain several links to redirect the user.
      - num_exclamations: The count of exclamation marks in the email.
                          An unusually high count may indicate spam.
      - email_length: The length of the email in characters. This can
                      help differentiate short, potentially templated spam emails
                      from longer legitimate communications.
      - sender_domain: The domain from which the email was sent
                       (e.g., gmail.com, yahoo.com, spamdomain.net). Some domains
                       are more frequently associated with spam.
      - email_client: The client or platform used to send or view
                      the email (e.g., Web, Mobile, Desktop, Other).
      - suspicion_score: A score (scale 1-5) indicating how
                         suspicious the email appears. Higher scores suggest greater
                         suspicion.
      - urgency_level: A score (scale 1-5) representing how
                       urgent the email appears. Spam emails may exaggerate urgency
                       to provoke a quick reaction.
      - subject: The subject line of the email. Spam emails often use
                 attention-grabbing phrases.
      - email_body: A snippet of the email body, which may include both
                    spam-like language and standard communication phrases.
      - target: The class label where 1 indicates a spam email and 0
                  indicates a non-spam email.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        random_state: Union[int, None] = 42,
        spam_ratio: float = 0.05,
    ) -> None:
        """
        Initialize the SpamDatasetGenerator instance.

        Args:
            n_samples (int): Number of samples to generate. Defaults to 1000.
            random_state (int): Seed for reproducibility. Defaults to 42.
            spam_ratio (float): Proportion of spam samples (target = 1) in the dataset.
                                Defaults to 0.05.
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.spam_ratio = spam_ratio

        np.random.seed(self.random_state)

        # Define an ordered categorical type for ordinal features (Likert scale 1-5)
        self.ordered_dtype = pd.api.types.CategoricalDtype(
            categories=[1, 2, 3, 4, 5], ordered=True
        )

    def print_description(self) -> None:
        """Prints the class description."""
        print(self.__doc__)

    def generate_features(self) -> pd.DataFrame:
        """
        Generate the feature set for the spam classification dataset without the target variable.

        Returns:
            pd.DataFrame: A DataFrame containing 9 features:
                - num_links, num_exclamations, email_length
                - sender_domain, email_client
                - suspicion_score, urgency_level
                - subject, email_body
        """
        # -----------------------------
        # Numeric Features
        # -----------------------------
        num_links: np.ndarray = np.random.poisson(lam=2, size=self.n_samples)
        num_exclamations: np.ndarray = np.random.poisson(lam=1, size=self.n_samples)
        email_length: np.ndarray = np.random.normal(
            loc=500, scale=100, size=self.n_samples
        ).astype(int)

        # -----------------------------
        # Categorical Features
        # -----------------------------
        sender_domain_pool: List[str] = [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "spamdomain.net",
            "outlook.com",
        ]
        sender_domain: np.ndarray = np.random.choice(
            sender_domain_pool, size=self.n_samples
        )

        email_client_pool: List[str] = ["Web", "Mobile", "Desktop", "Other"]
        email_client: np.ndarray = np.random.choice(
            email_client_pool, size=self.n_samples
        )

        # -----------------------------
        # Ordered Features
        # -----------------------------
        suspicion_score_raw: np.ndarray = np.random.randint(1, 6, size=self.n_samples)
        urgency_level_raw: np.ndarray = np.random.randint(1, 6, size=self.n_samples)
        suspicion_score: pd.Series = pd.Series(suspicion_score_raw).astype(
            self.ordered_dtype
        )
        urgency_level: pd.Series = pd.Series(urgency_level_raw).astype(
            self.ordered_dtype
        )

        # -----------------------------
        # Text Features
        # -----------------------------
        subject_pool: List[str] = [
            "Limited time offer",
            "Meeting reminder",
            "Win a free iPhone",
            "Project update",
            "You have been selected",
            "Invoice attached",
            "Urgent action required",
            "Congratulations, you've won",
            "Team outing",
            "Claim your prize now",
        ]
        subject: np.ndarray = np.random.choice(subject_pool, size=self.n_samples)

        email_body_pool: List[str] = [
            "Congratulations, you have been selected for an exclusive deal.",
            "Please find attached the minutes of today's meeting.",
            "Don't miss out on this once in a lifetime opportunity!",
            "Your account has been updated successfully.",
            "Claim your prize now by clicking the link below.",
            "The project deadline has been moved up.",
            "Urgent: Your action is required immediately.",
            "Let's catch up over coffee next week.",
            "This is a reminder for your upcoming appointment.",
            "Limited time discount available only today!",
        ]
        email_body: np.ndarray = np.random.choice(email_body_pool, size=self.n_samples)

        # -----------------------------
        # Assemble the Features
        # -----------------------------
        X: pd.DataFrame = pd.DataFrame(
            {
                "num_links": num_links,
                "num_exclamations": num_exclamations,
                "email_length": email_length,
                "sender_domain": sender_domain,
                "email_client": email_client,
                "suspicion_score": suspicion_score,
                "urgency_level": urgency_level,
                "subject": subject,
                "email_body": email_body,
            }
        )
        return X

    def generate_target(
        self, X: pd.DataFrame, na_ratio_for_non_spam: float = 0.0
    ) -> pd.Series:
        """
        Generate the binary target variable for the spam classification dataset using a calibrated intercept.

        Args:
            na_ratio_for_non_spam (float): The proportion of non-spam samples (target = 0) to set to pd.NA.
                                           Must be <= 1.0. Defaults to 0.0.
            X (pd.DataFrame): The feature dataset.

        Returns:
            pd.Series: A Series containing the binary target variable.
        """
        desired = self.spam_ratio

        email_length_scaled = (X["email_length"].values - 500) / 100.0

        X_model = np.column_stack(
            [
                X["num_links"].values,
                X["num_exclamations"].values,
                email_length_scaled,
                X["suspicion_score"].astype(int).values,
                X["urgency_level"].astype(int).values,
            ]
        )

        # Add a bias term
        X_model_bias = np.column_stack([np.ones(self.n_samples), X_model])

        # Draw from normal prior
        self.beta = np.random.normal(loc=0, scale=1, size=X_model_bias.shape[1])
        z = np.dot(X_model_bias, self.beta)

        # Define a function to calibrate the intercept shift 'c'
        def f(c):
            return np.mean(expit(z + c)) - desired

        # Calibrate 'constant' by root-finding, solve for c
        sol = root_scalar(f, bracket=[-20, 20], method="bisect")
        self.calib_intercept = sol.root

        # Adjust linear predictor and compute probabilities.
        z_adjusted = z + self.calib_intercept

        # Sigmoid (inverse logit/expit) function
        p = 1 / (1 + np.exp(-z_adjusted))

        # print("Mean probability after calibration:", np.mean(p))
        # print("Beta coefficients:", self.beta)
        # print("Calibrated intercept (c):", self.calib_intercept)

        # Draw samples from Bernoulli likelihood
        y = np.random.binomial(1, p, size=self.n_samples)

        target_series = pd.Series(y, name="target")

        # Create semi-supervised / positive unlabelled
        # ---------------------------------------------
        # Set a fraction of non-spam samples to pd.NA
        if na_ratio_for_non_spam > 0:
            assert na_ratio_for_non_spam <= 1.0, "na_ratio_for_non_spam must be <= 1.0"
            non_spam_indices = target_series[target_series == 0].index
            num_to_replace = int(len(non_spam_indices) * na_ratio_for_non_spam)
            indices_to_replace = np.random.choice(
                non_spam_indices, size=num_to_replace, replace=False
            )
            target_series.loc[indices_to_replace] = pd.NA

        target_series = target_series.astype("Int64")
        return target_series

    def load_dataset(
        self, na_ratio_for_non_spam: float = 0.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate the complete dataset (features and target) for spam classification.

        Args:
            na_ratio_for_non_spam (float): The proportion of non-spam samples (target = 0) to set to pd.NA.
                                           Defaults to 0.0.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Feature DataFrame with 9 columns.
                - y (pd.Series): Target variable Series.
        """
        X = self.generate_features()
        # X = pd.DataFrame(np.random.randn(self.n_samples, 5))
        y = self.generate_target(X, na_ratio_for_non_spam)
        return X, y
