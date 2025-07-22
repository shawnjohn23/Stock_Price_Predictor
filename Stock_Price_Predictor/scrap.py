# main.py

from stock_data_loader import download_stock_data
from stock_data_loader import load_data
from sequence_builder import scale_data, create_sequences
import shap

explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_val[:100])
shap.summary_plot(shap_values, X_val[:100], feature_names=scaled_df.columns)

"""





# Display all columns (overrides truncation)
pd.set_option('display.max_columns', None)

# Show basic info
print("🔍 Dataset Info:")
print(scaled_df.info())

# Show first few rows
print("\n📌 Sample Rows:")
print(scaled_df.head(3))  # Adjust the number as needed

# Show descriptive stats
print("\n📊 Summary Stats:")
print(scaled_df.describe())

"""