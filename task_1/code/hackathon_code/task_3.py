import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from utils import Utils

def plotCorrelationGraph(df: pd.DataFrame, correlation_matrix: pd.DataFrame):
    # Get the features with the highest correlation with y
    best_features = ['D1', 'P1', 'advanced_order_days']

    # Create a bar plot of the correlation coefficients of the best features
    fig = go.Figure()
    for feature in df.columns:
        if feature in ('is_canceled', 'D4', 'P4'):
            continue
        if feature in best_features:
            fig.add_trace(go.Bar(
                x=[feature],
                y=[correlation_matrix[feature]['is_canceled']],
                marker_color='red'
            ))
        else:
            fig.add_trace(go.Bar(
                x=[feature],
                y=[correlation_matrix[feature]['is_canceled']],
                marker_color='blue'
            ))

    fig.update_layout(
        title='Correlation of Best Features with is_canceled',
        xaxis_title='<b>Feature</b>',
        yaxis_title='<b>Correlation</b>',
        showlegend=False
    )

    fig.show()


def plotAverageValues(df):
    grouped_data = df.groupby('is_canceled')['D1', 'P1', 'advanced_order_days'].mean().reset_index()

    # Create the grouped bar plot
    bar_plot = go.Figure(data=[
        go.Bar(name='D1', x=grouped_data['is_canceled'], y=grouped_data['D1']),
        go.Bar(name='P1', x=grouped_data['is_canceled'], y=grouped_data['P1']),
        go.Bar(name='advanced_order_days', x=grouped_data['is_canceled'], y=grouped_data['advanced_order_days'])
    ])

    # Customize the layout
    bar_plot.update_layout(
        title='Average Values of Selected Features for is_canceled',
        xaxis=dict(title='is_canceled'),
        yaxis=dict(title='Average Value'),
        barmode='group'
    )

    # Show the bar plot
    bar_plot.show()


def plotThreeFeaturesTest(df: pd.DataFrame, y: pd.DataFrame):
    features_to_keep = ['D1', 'P1', 'advanced_order_days']

    # Drop the unwanted features from the DataFrame
    #df = df.drop(columns=['is_canceled'], axis=1) #original data
    #df = df.drop(columns=[col for col in df.columns if col not in features_to_keep]) #three features only
    df = df.drop(columns=['D1', 'P1', 'advanced_order_days', 'is_canceled'], axis=1) #without features

    train_f1_scores, test_f1_scores = [], []
    train_X, test_X, train_y, test_y = train_test_split(df, y, test_size=0.3)

    for percentage in range(0, 100, 9):
        n_samples = int((percentage / 100) * len(train_y))

        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
        X_train_subset = train_X[:n_samples]
        y_train_subset = train_y[:n_samples]
        xgb_model.fit(X_train_subset, y_train_subset)
        y_train_pred = xgb_model.predict(train_X)
        y_test_pred = xgb_model.predict(test_X)

        # Calculate the F1 scores
        train_f1 = f1_score(train_y, y_train_pred, average='macro')
        test_f1 = f1_score(test_y, y_test_pred, average='macro')

        # Append the scores to the lists
        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)

    #plt.plot(range(0, 100, 9), train_f1_scores, label='Train f1_score')
    plt.plot(range(0, 100, 9), test_f1_scores, label='Test f1_score')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('f1_score')
    plt.title('xgBoost ensembles Performance without the three features')
    plt.legend()

    plt.show()

def plotBestFeatures(X: pd.DataFrame, y: pd.DataFrame):
    # Assuming X is your feature matrix DataFrame and y is your target vector
    df = pd.concat([X, y], axis=1)
    correlation_matrix = df.corr()
    plotCorrelationGraph(df, correlation_matrix)
    plotAverageValues(df)
    plotThreeFeaturesTest(df, y)