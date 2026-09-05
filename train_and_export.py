import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def main():
    print("1. Loading raw Telco dataset...")
    df = pd.read_excel(r"Telco_customer_churn.xlsx")
    
    # Save a copy with CustomerID for UI customer lookup if available
    customer_ids = df["CustomerID"].copy() if "CustomerID" in df.columns else None

    # Replicate drop columns from notebook
    drop_cols = [
        'CustomerID', 'Country', 'State', 'Count', 'Zip Code', 'Lat Long', 
        'Churn Score', 'Churn Label', 'Churn Reason', 'Year',
        'Multiple Lines', 'Senior Citizen', 'Gender', 'Phone Service', 
        'Device Protection', 'Total Charges', 'Monthly Charges', 'Latitude', 
        'Longitude', 'CLTV', 'Online Backup', 'Partner'
    ]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    x = df_clean.drop(columns=['Churn Value'])
    y = df_clean['Churn Value']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    object_cols = list(x.select_dtypes(include='object').columns)
    num_cols = list(x.select_dtypes(include='number').columns)

    print("2. Building preprocessing and XGBoost pipeline...")
    op1 = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), object_cols),
            ('num', StandardScaler(), num_cols)
        ],
        remainder='passthrough'
    )

    # Use optimal parameters
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    smote = SMOTE(random_state=42)

    pipeline = Pipeline([
        ('preprocessor1', op1),
        ('smote', smote),
        ('model', xgb)
    ])

    print("3. Training pipeline on training data...")
    pipeline.fit(x_train, y_train)

    preprocessor = pipeline.named_steps['preprocessor1']
    model = pipeline.named_steps['model']

    raw_feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [
        f.replace('ohe__', '').replace('num__', '').replace('remainder__', '')
        for f in raw_feature_names
    ]

    print("4. Precomputing test transformations and SHAP values...")
    x_test_transformed = pd.DataFrame(
        preprocessor.transform(x_test),
        columns=clean_feature_names,
        index=x_test.index
    )

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(x_test_transformed)
    shap_values = explainer.shap_values(x_test_transformed)

    # Calculate test predictions & probabilities
    y_test_pred = model.predict(x_test_transformed)
    y_test_proba = model.predict_proba(x_test_transformed)[:, 1]

    # Create an enriched test dataframe for the dashboard
    test_df_enriched = x_test.copy()
    if customer_ids is not None:
        test_df_enriched['CustomerID'] = customer_ids.loc[x_test.index]
    test_df_enriched['Actual Churn'] = y_test
    test_df_enriched['Predicted Churn'] = y_test_pred
    test_df_enriched['Churn Probability'] = np.round(y_test_proba, 4)
    test_df_enriched['Risk Tier'] = pd.cut(
        test_df_enriched['Churn Probability'],
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=['Low Risk', 'Moderate Risk', 'High Risk']
    )

    # Extract unique values for UI dropdowns
    categorical_options = {col: sorted(list(x[col].dropna().unique())) for col in object_cols}
    numerical_ranges = {
        col: {
            'min': float(x[col].min()),
            'max': float(x[col].max()),
            'default': float(x[col].median())
        } for col in num_cols
    }

    print("5. Saving artifacts...")
    artifacts = {
        'pipeline': pipeline,
        'preprocessor': preprocessor,
        'model': model,
        'clean_feature_names': clean_feature_names,
        'categorical_options': categorical_options,
        'numerical_ranges': numerical_ranges,
        'x_test': x_test,
        'y_test': y_test,
        'x_test_transformed': x_test_transformed,
        'test_df_enriched': test_df_enriched,
        'shap_values': shap_values,
        'expected_value': explainer.expected_value
    }

    joblib.dump(artifacts, "churn_model_assets.joblib")
    print("Exported successfully to churn_model_assets.joblib!")

if __name__ == "__main__":
    main()
