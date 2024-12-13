import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


df = pd.read_csv("uber.csv")


Q1 = df['DeliveryDistance'].quantile(0.25)
Q3 = df['DeliveryDistance'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['DeliveryDistance'] >= (Q1 - 1.5 * IQR)) & (df['DeliveryDistance'] <= (Q3 + 1.5 * IQR))]

# Age distribution analysis
bins = [20, 30, 40]  # age group ranges
labels = ['20-30', '31-40']

df['Age Group'] = pd.cut(df['Delivery_person_Age'], bins=bins, labels=labels, right=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x='multiple_deliveries', y='Time_taken(min)', hue='Age Group', ax=axes[0], palette='cool')
axes[0].set_title('Delivery Time vs. Multiple Deliveries by Age Groups')
axes[0].set_xlabel('Multiple Deliveries')
axes[0].set_ylabel('Time Taken (min)')

sns.lineplot(data=df, x='multiple_deliveries', y='Time_taken(min)', hue='Age Group', ax=axes[1], marker='o', palette='cool')
axes[1].set_title('Delivery Time vs. Multiple Deliveries by Age Groups')
axes[1].set_xlabel('Multiple Deliveries')
axes[1].set_ylabel('Time Taken (min)')

plt.tight_layout()
plt.show()

# label encoder
label_encoder = LabelEncoder()

#columns to be label-encoded
columns_to_encode = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order',
                     'Type_of_vehicle', 'Festival', 'City', 'Order_Day', 'TypeOfMeal']

for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

high_traffic_df = df[df['Road_traffic_density'] == 0]

age_vehicle_analysis = high_traffic_df.groupby(['Delivery_person_Age', 'Type_of_vehicle']).agg({
    'Time_taken(min)': ['mean', 'count']
}).reset_index()

age_vehicle_analysis.columns = ['Delivery_person_Age', 'Type_of_vehicle', 'Avg_DeliveryTime', 'Order_Count']

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x='Delivery_person_Age',
    y='Avg_DeliveryTime',
    hue='Type_of_vehicle',
    data=age_vehicle_analysis,
    palette={0: 'yellow', 1: 'purple'},
    style='Type_of_vehicle',
    markers={0: 'o', 1: 's'},
    edgecolor='black'
)

#Cluster 1: Young drivers (ages 20-30) with delivery times
ellipse1 = mpatches.Ellipse((25, 24), width=14, height=10, angle=0, color='blue', fill=False, linewidth=2, label="Young, Fast Delivery")
scatter.add_patch(ellipse1)

#Cluster 2: Middle-aged drivers (ages 30-40) with delivery times
ellipse2 = mpatches.Ellipse((35, 30), width=12, height=9, angle=0, color='orange', fill=False, linewidth=2, label="Middle-aged, Moderate Delivery")
scatter.add_patch(ellipse2)

# Custom legend for vehicle types
legend_labels = {
    0: "Motorcycle",
    1: "Scooter"
}
handles = [mpatches.Patch(color=color, label=legend_labels[key]) for key, color in {0: 'yellow', 1: 'purple'}.items()]

handles.append(ellipse1)
handles.append(ellipse2)

plt.legend(handles=handles, title='Vehicle Type and Age Cluster', loc='upper right')
plt.title('Impact of Delivery Person Age and Vehicle Type on Delivery Time in High Traffic Density (Clustered)')
plt.xlabel('Delivery Person Age')
plt.ylabel('Average Delivery Time (min)')
plt.tight_layout()
plt.show()

correlation_matrix = df[
    ['Time_taken(min)', 'DeliveryDistance', 'Delivery_person_Age', 'Delivery_person_Ratings', 'Weatherconditions',
       'Road_traffic_density', 'Vehicle_condition', 'multiple_deliveries', 'OrderTime' ]].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

#numerical feature scaling
numerical_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
                      'multiple_deliveries', 'DeliveryDistance', 'OrderTime']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
df.head()
df = df.drop_duplicates()
df.head()

# features for KNN
X = df[['Delivery_person_Age', 'Delivery_person_Ratings', 'Weatherconditions',
       'Road_traffic_density', 'Vehicle_condition', 'DeliveryDistance', 'multiple_deliveries', 'OrderTime']]
y = df['Time_taken(min)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Regression
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate KNN
y_pred_knn = knn.predict(X_test)
print(f'KNN Mean Absolute Error: {mean_absolute_error(y_test, y_pred_knn)}')
print(f'KNN Mean Squared Error: {mean_squared_error(y_test, y_pred_knn)}')

# Decision Tree Regressor
dtree = DecisionTreeRegressor(random_state=42)
dtree.fit(X_train, y_train)
importance_dtree = dtree.feature_importances_

y_pred_dtree = dtree.predict(X_test)
print(f'Decision Tree Mean Squared Error: {mean_squared_error(y_test, y_pred_dtree)}')

# Random Forest Regressor
rforest = RandomForestRegressor(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)
importance_rf = rforest.feature_importances_


y_pred_rf = rforest.predict(X_test)
print(f'Random Forest Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}')

#feature importance in a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Decision Tree': importance_dtree,
    'Random Forest': importance_rf
})


plt.figure(figsize=(12, 8))
feature_importance_df.set_index('Feature').plot(kind='bar', figsize=(12, 8))
plt.title("Feature Importance Comparison (Decision Tree and Random Forest)")
plt.ylabel("Feature Importance")
plt.xlabel("Features")
plt.legend(title="Model")
plt.tight_layout()
plt.show()
# Model Evaluation: Actual vs Predicted for KNN, Decision Tree, and Random Forest

#DataFrame to store actual and predicted values
evaluation_df = pd.DataFrame({
    'Actual': y_test,
    'KNN_Predicted': y_pred_knn,
    'DecisionTree_Predicted': y_pred_dtree,
    'RandomForest_Predicted': y_pred_rf
})

# Actual vs Predicted for each model
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(evaluation_df['Actual'], evaluation_df['KNN_Predicted'], color='blue', alpha=0.6)
plt.plot([evaluation_df['Actual'].min(), evaluation_df['Actual'].max()],
         [evaluation_df['Actual'].min(), evaluation_df['Actual'].max()], color='red', lw=2)
plt.title('KNN: Actual vs Predicted')
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')

plt.subplot(1, 3, 2)
plt.scatter(evaluation_df['Actual'], evaluation_df['DecisionTree_Predicted'], color='green', alpha=0.6)
plt.plot([evaluation_df['Actual'].min(), evaluation_df['Actual'].max()],
         [evaluation_df['Actual'].min(), evaluation_df['Actual'].max()], color='red', lw=2)
plt.title('Decision Tree: Actual vs Predicted')
plt.xlabel('Actual Delivery Time')

plt.subplot(1, 3, 3)
plt.scatter(evaluation_df['Actual'], evaluation_df['RandomForest_Predicted'], color='orange', alpha=0.6)
plt.plot([evaluation_df['Actual'].min(), evaluation_df['Actual'].max()],
         [evaluation_df['Actual'].min(), evaluation_df['Actual'].max()], color='red', lw=2)
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual Delivery Time')

plt.tight_layout()
plt.show()

# Comparing Model Performance using MAE and MSE
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)

mae_dtree = mean_absolute_error(y_test, y_pred_dtree)
mse_dtree = mean_squared_error(y_test, y_pred_dtree)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

#DataFrame to store evaluation metrics
performance_df = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Random Forest'],
    'MAE': [mae_knn, mae_dtree, mae_rf],
    'MSE': [mse_knn, mse_dtree, mse_rf]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=performance_df,hue='Model', palette='coolwarm', legend=False)
plt.title('Mean Squared Error (MSE) Comparison')
plt.show()







