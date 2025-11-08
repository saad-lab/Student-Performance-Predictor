
# Student Performance Prediction System
# Complete implementation including data preprocessing, model training, evaluation and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os

# Set the font sizes
SMALL_SIZE = 6
MEDIUM_SIZE = 7
TITLE_SIZE = 8
XAXIS_SIZE = 4  # Extra small size for x-axis labels

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the student data
    """
    print("Loading and preprocessing data...")
    # Load the dataset
    df = pd.read_csv(file_path)

    # Select features for modeling
    features_to_use = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                      'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week',
                      'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level',
                      'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
    target = 'Grade'

    # Check for missing values
    missing_values = df[features_to_use + [target]].isnull().sum()
    print("Missing values in selected features:")
    print(missing_values)

    # Identify students with excessive missing data (more than 30% of features)
    missing_threshold = 0.3 * len(features_to_use)
    students_with_missing = df[df[features_to_use].isnull().sum(axis=1) > missing_threshold]
    print(f"\nStudents with excessive missing data (>{missing_threshold} features missing):")
    print(students_with_missing[['Student_ID', 'First_Name', 'Last_Name']])

    return df, features_to_use, target

def handle_missing_values(df, features_to_use, method='impute'):
    """
    Handle missing values using either imputation or dropping
    """
    if method == 'impute':
        print("\nHandling missing values using imputation...")
        # For numerical features
        numerical_features = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                             'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week',
                             'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
        imputer_num = SimpleImputer(strategy='median')
        df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

        # For categorical features
        categorical_features = ['Extracurricular_Activities', 'Internet_Access_at_Home', 
                               'Parent_Education_Level', 'Family_Income_Level']
        for feature in categorical_features:
            mode_value = df[feature].mode()[0]
            df[feature].fillna(mode_value, inplace=True)

        return df, numerical_features, categorical_features

    elif method == 'drop':
        print("\nHandling missing values by dropping rows...")
        df_dropped = df.dropna(subset=features_to_use)
        print(f"Total rows before dropping: {len(df)}")
        print(f"Total rows after dropping: {len(df_dropped)}")
        print(f"Rows dropped: {len(df) - len(df_dropped)} ({(len(df) - len(df_dropped))/len(df)*100:.2f}%)")

        # Define numerical and categorical features
        numerical_features = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                             'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week',
                             'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
        categorical_features = ['Extracurricular_Activities', 'Internet_Access_at_Home', 
                               'Parent_Education_Level', 'Family_Income_Level']

        return df_dropped, numerical_features, categorical_features

    elif method == 'drop_features':
        print("\nHandling missing values by dropping problematic features...")
        # Identify features with significant missing values
        print("Features with significant missing values:")
        print(df[['Attendance (%)', 'Assignments_Avg', 'Parent_Education_Level']].isnull().sum())

        # Select features, excluding the problematic ones with missing values
        features_to_use_reduced = ['Midterm_Score', 'Final_Score', 
                                  'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week',
                                  'Extracurricular_Activities', 'Internet_Access_at_Home', 
                                  'Family_Income_Level', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']

        # Define numerical and categorical features
        numerical_features = ['Midterm_Score', 'Final_Score', 
                             'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week',
                             'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
        categorical_features = ['Extracurricular_Activities', 'Internet_Access_at_Home', 
                               'Family_Income_Level']

        return df, numerical_features, categorical_features

    else:
        raise ValueError("Method must be one of 'impute', 'drop', or 'drop_features'")

def prepare_features(df, numerical_features, categorical_features):
    """
    Prepare features for modeling by encoding categorical variables
    """
    print("\nPreparing features for modeling...")
    # Encode categorical features
    categorical_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

    # Combine features
    X = pd.concat([df[numerical_features], categorical_encoded], axis=1)
    y = df['Grade']

    return X, y

def train_and_evaluate_model(X, y, n_estimators=125):
    """
    Train and evaluate a Random Forest model
    """
    print(f"\nTraining Random Forest model with {n_estimators} trees...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Random Forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return rf, X_train, X_test, y_train, y_test, y_pred, accuracy

def create_visualizations(rf, X, X_test, y_test, y_pred):
    """
    Create visualizations for model evaluation and interpretation
    """
    print("\nCreating visualizations...")
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("Feature Importance:")
    print(feature_importance.head(10))

    # Set font sizes for matplotlib
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=TITLE_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=XAXIS_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=TITLE_SIZE)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

    # SHAP-style Feature Importance
    plt.figure(figsize=(10, 7))

    # Create a horizontal bar chart with colored bars to mimic SHAP style
    colors = plt.cm.tab10(np.linspace(0, 1, 5))  # 5 classes for grades A-F

    # Create the horizontal bar chart
    feature_importance_sorted = feature_importance.sort_values('Importance', ascending=True)
    bars = plt.barh(y=feature_importance_sorted['Feature'], width=feature_importance_sorted['Importance'], 
           color=colors[0], height=0.7)

    # Set the x-axis ticks with specific format and very small font
    plt.xticks([-0.25, 0, 0.25], ['-0.25', '0', '0.25'], fontsize=XAXIS_SIZE)

    # Add a two-line title with proper spacing
    plt.suptitle("SHAP Feature Importance", fontsize=TITLE_SIZE, y=0.98)
    plt.title("Impact on Model Output", fontsize=SMALL_SIZE, pad=10)

    # Set labels
    plt.xlabel('mean(|SHAP value|)', fontsize=SMALL_SIZE)
    plt.ylabel('Feature', fontsize=SMALL_SIZE)

    # Adjust margins
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the figure with high resolution
    plt.savefig('shap_style_feature_importance.png', dpi=300, bbox_inches='tight')

    # Try to create a proper SHAP plot if the library is available
    try:
        import shap

        # Create a small sample for SHAP analysis
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(rf)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Create SHAP summary plot with bar type (feature importance)
        plt.figure(figsize=(10, 7))

        # Create the SHAP summary plot
        shap.summary_plot(shap_values, X_sample, plot_type="bar", class_names=sorted(y_test.unique()), show=False)

        # Get current axes and adjust font sizes
        ax = plt.gca()

        # Clear the current title
        ax.set_title("")

        # Add a two-line title with proper spacing
        plt.suptitle("SHAP Feature Importance", fontsize=TITLE_SIZE, y=0.98)
        plt.title("Impact on Model Output", fontsize=SMALL_SIZE, pad=10)

        # Adjust x-axis label
        ax.set_xlabel('mean(|SHAP value|)', fontsize=SMALL_SIZE)

        # Manually set x-ticks to use "0" instead of "0.00"
        ax.set_xticks([-0.25, 0, 0.25])
        ax.set_xticklabels(['-0.25', '0', '0.25'], fontsize=XAXIS_SIZE)

        # Adjust y-tick labels
        plt.yticks(fontsize=SMALL_SIZE)

        # Add more space at the top
        plt.subplots_adjust(top=0.88)

        # Save the figure with high resolution
        plt.tight_layout()
        plt.savefig('real_shap_feature_importance.png', dpi=300, bbox_inches='tight')

        print("Created SHAP visualization successfully.")

    except Exception as e:
        print(f"Error creating SHAP plot: {e}")
        print("Using alternative feature importance visualization.")

    print("Visualizations saved to disk.")

    return feature_importance

def compare_tree_counts(X, y):
    """
    Compare model performance with different numbers of trees
    """
    print("\nComparing model performance with different numbers of trees...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Try with different tree counts
    tree_counts = [50, 100, 125, 150, 200, 250]
    accuracies = []

    for n_trees in tree_counts:
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Accuracy with {n_trees} trees: {acc:.4f}")

    # Plot accuracy vs number of trees
    plt.figure(figsize=(10, 6))
    plt.plot(tree_counts, accuracies, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Number of Trees')
    plt.grid(True)
    plt.savefig('accuracy_vs_trees.png')

    # Create a summary table
    summary = {
        'Number of Trees': tree_counts,
        'Accuracy': accuracies,
        'Difference from 100 trees': [acc - accuracies[1] for acc in accuracies]
    }
    summary_df = pd.DataFrame(summary)
    print("\nSummary Table:")
    print(summary_df)

    # Save summary to CSV
    summary_df.to_csv('tree_count_comparison.csv', index=False)

    return summary_df

def check_for_bias(rf, X_test, y_test, df):
    """
    Check for potential bias in model predictions
    """
    print("\nChecking for potential bias in predictions:")
    # Predict on test set
    y_pred = rf.predict(X_test)

    # Analyze model performance across different demographic groups
    if 'Gender' in df.columns:
        gender_performance = {}
        for gender in df['Gender'].unique():
            gender_mask = df['Gender'] == gender
            gender_indices = df[gender_mask].index
            test_indices = X_test.index.intersection(gender_indices)
            if len(test_indices) > 0:
                X_gender = X_test.loc[test_indices]
                y_gender = y_test.loc[test_indices]
                y_gender_pred = rf.predict(X_gender)
                gender_acc = accuracy_score(y_gender, y_gender_pred)
                gender_performance[gender] = gender_acc

        print("Model accuracy by gender:")
        for gender, acc in gender_performance.items():
            print(f"{gender}: {acc:.4f}")

    # Analyze model performance across income levels
    if 'Family_Income_Level' in df.columns:
        income_performance = {}
        for income in df['Family_Income_Level'].unique():
            income_mask = df['Family_Income_Level'] == income
            income_indices = df[income_mask].index
            test_indices = X_test.index.intersection(income_indices)
            if len(test_indices) > 0:
                X_income = X_test.loc[test_indices]
                y_income = y_test.loc[test_indices]
                y_income_pred = rf.predict(X_income)
                income_acc = accuracy_score(y_income, y_income_pred)
                income_performance[income] = income_acc

        print("\nModel accuracy by family income level:")
        for income, acc in income_performance.items():
            print(f"{income}: {acc:.4f}")

    return gender_performance if 'Gender' in df.columns else None, income_performance if 'Family_Income_Level' in df.columns else None

def generate_recommendations(feature_importance):
    """
    Generate recommendations based on feature importance
    """
    print("\nRecommendations based on feature importance:")
    for feature, importance in feature_importance.head(5).values:
        print(f"- Focus on {feature} (Importance: {importance:.4f})")

    return feature_importance.head(5)

def create_report(rf, X_test, y_test, y_pred, feature_importance):
    """
    Create a summary report of model performance
    """
    print("\nCreating model performance summary...")
    # Save the model performance metrics to a file
    performance_summary = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Feature Importance': feature_importance.to_dict(),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }

    import json
    with open('model_performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=4)

    print("Model performance summary saved to 'model_performance_summary.json'")

    return performance_summary

def main():
    """
    Main function to run the complete analysis
    """
    # Load and preprocess data
    df, features_to_use, target = load_and_preprocess_data('Students_Grading_Dataset.csv')

    # Handle missing values (choose method: 'impute', 'drop', or 'drop_features')
    df_processed, numerical_features, categorical_features = handle_missing_values(df, features_to_use, method='impute')

    # Prepare features
    X, y = prepare_features(df_processed, numerical_features, categorical_features)

    # Train and evaluate model
    rf, X_train, X_test, y_train, y_test, y_pred, accuracy = train_and_evaluate_model(X, y, n_estimators=125)

    # Create visualizations
    feature_importance = create_visualizations(rf, X, X_test, y_test, y_pred)

    # Compare different tree counts
    tree_comparison = compare_tree_counts(X, y)

    # Check for bias
    gender_performance, income_performance = check_for_bias(rf, X_test, y_test, df)

    # Generate recommendations
    recommendations = generate_recommendations(feature_importance)

    # Create summary report
    performance_summary = create_report(rf, X_test, y_test, y_pred, feature_importance)

    print("\nAnalysis complete!")
    print(f"Final model accuracy: {accuracy:.4f}")
    print("Top 5 important features:")
    print(feature_importance.head(5))

if __name__ == "__main__":
    main()
